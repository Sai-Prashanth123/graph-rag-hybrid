from typing import List, Dict, Any
from langchain_core.documents import Document
from neo4j import GraphDatabase

from config import RAGConfig


class GraphStore:
    def __init__(self, config: RAGConfig):
        if not config.NEO4J_URI:
            raise ValueError("NEO4J_URI is empty; cannot connect to Neo4j.")
        self.config = config
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def add_chunks(self, docs: List[Document]) -> None:
        """Write each document as a Chunk node, idempotent via MERGE."""
        if not docs:
            return
        with self.driver.session(database=self.config.NEO4J_DATABASE) as session:
            for d in docs:
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text,
                        c.doc_name = $doc_name,
                        c.chunk_index = $chunk_index
                    """,
                    chunk_id=d.metadata.get("chunk_id"),
                    text=d.page_content,
                    doc_name=d.metadata.get("name", ""),
                    chunk_index=d.metadata.get("chunk_index", 0),
                )

    def add_entities(self, triples: Dict[str, Any]) -> None:
        """
        Write extracted entities & relations.

        triples = {
            "chunk_id": str,
            "entities": [{"name": str, "type": str}, ...],
            "relations": [{"source": str, "target": str, "type": str}, ...],
        }
        """
        chunk_id = triples.get("chunk_id")
        entities = triples.get("entities") or []
        relations = triples.get("relations") or []
        if not chunk_id:
            return

        with self.driver.session(database=self.config.NEO4J_DATABASE) as session:
            for ent in entities:
                name = (ent.get("name") or "").strip()
                etype = (ent.get("type") or "Concept").strip()
                if not name:
                    continue
                session.run(
                    """
                    MERGE (e:Entity {name_lower: toLower($name)})
                    SET e.name = $name, e.type = $type
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    name=name, type=etype, chunk_id=chunk_id,
                )

            for rel in relations:
                src = (rel.get("source") or "").strip()
                tgt = (rel.get("target") or "").strip()
                rtype = (rel.get("type") or "RELATES_TO").strip().upper().replace(" ", "_")
                if not src or not tgt:
                    continue
                session.run(
                    """
                    MERGE (a:Entity {name_lower: toLower($src)})
                    MERGE (b:Entity {name_lower: toLower($tgt)})
                    MERGE (a)-[r:RELATION {type: $rtype}]->(b)
                    """,
                    src=src, tgt=tgt, rtype=rtype,
                )

    def search(self, entity_names: List[str], k: int = 10, hops: int = 1) -> List[Document]:
        """Find chunks related to the given entity names. hops 1 or 2."""
        if not entity_names:
            return []

        names_lower = [n.lower() for n in entity_names]

        if hops <= 1:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $names
               OR any(n IN $names WHERE toLower(e.name) CONTAINS n)
            WITH collect(e) AS matched
            UNWIND matched AS e
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.id AS id, c.text AS text,
                   c.doc_name AS doc_name, c.chunk_index AS chunk_index,
                   count(*) AS relevance
            ORDER BY relevance DESC
            LIMIT $k
            """
        else:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $names
               OR any(n IN $names WHERE toLower(e.name) CONTAINS n)
            WITH collect(e) AS matched
            UNWIND matched AS e
            MATCH (e)-[r:RELATION]-(neighbor:Entity)-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.id AS id, c.text AS text,
                   c.doc_name AS doc_name, c.chunk_index AS chunk_index,
                   count(DISTINCT neighbor) AS relevance
            ORDER BY relevance DESC
            LIMIT $k
            """

        with self.driver.session(database=self.config.NEO4J_DATABASE) as session:
            records = session.run(query, names=names_lower, k=k)
            results: List[Document] = []
            for r in records:
                results.append(Document(
                    page_content=r["text"] or "",
                    metadata={
                        "chunk_id": r["id"],
                        "name": r["doc_name"] or "",
                        "chunk_index": r["chunk_index"] or 0,
                        "graph_relevance": r["relevance"],
                    },
                ))
            return results
