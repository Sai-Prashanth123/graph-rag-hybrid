import asyncio
import json
import re
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config import RAGConfig

if TYPE_CHECKING:
    from graph_store import GraphStore


_PROMPT_TEMPLATE = """Extract entities and relationships from the text below.

Entity types (use ONLY these):
  {entity_types}

Return JSON only (no prose, no code fences):
{{
  "entities": [{{"name": "...", "type": "<one of the types above>"}}],
  "relations": [{{"source": "...", "target": "...", "type": "VERB_PHRASE"}}]
}}

Text:
{text}
"""


class GraphExtractor:
    def __init__(self, llm: Any, config: RAGConfig):
        self.llm = llm
        self.config = config

    def _truncate(self, text: str) -> str:
        return text[: self.config.EXTRACTION_MAX_CHARS]

    def _parse(self, raw: str) -> Dict[str, List[Dict[str, Any]]]:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"entities": [], "relations": []}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
        return {
            "entities": data.get("entities", []) or [],
            "relations": data.get("relations", []) or [],
        }

    def _build_prompt(self, text: str) -> str:
        return _PROMPT_TEMPLATE.format(
            entity_types=", ".join(self.config.ENTITY_TYPES),
            text=self._truncate(text),
        )

    async def _extract_one(self, doc: Document) -> Dict[str, Any]:
        prompt = self._build_prompt(doc.page_content)
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            triples = self._parse(content)
        except Exception as e:
            print(f"Warning: extraction failed for {doc.metadata.get('chunk_id')}: {e}")
            triples = {"entities": [], "relations": []}
        triples["chunk_id"] = doc.metadata.get("chunk_id")
        return triples

    async def extract_and_store_async(
        self, docs: List[Document], graph_store: "GraphStore"
    ) -> None:
        sem = asyncio.Semaphore(self.config.EXTRACTION_CONCURRENCY)

        async def bound_extract(d: Document) -> Dict[str, Any]:
            async with sem:
                return await self._extract_one(d)

        tasks = [bound_extract(d) for d in docs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for triples in results:
            graph_store.add_entities(triples)

    def extract_and_store(self, docs: List[Document], graph_store: "GraphStore") -> None:
        asyncio.run(self.extract_and_store_async(docs, graph_store))

    def _extract_entity_names(self, question: str) -> List[str]:
        """Synchronously ask the LLM to list entities in a question (used at query time)."""
        prompt = (
            "Extract entity names from this question. "
            "Return JSON only: {\"entities\": [\"name1\", \"name2\"]}\n\n"
            f"Question: {self._truncate(question)}"
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return []
            data = json.loads(m.group(0))
            names = data.get("entities", []) or []
            return [n for n in names if isinstance(n, str) and n.strip()]
        except Exception as e:
            print(f"Warning: question entity extraction failed: {e}")
            return []
