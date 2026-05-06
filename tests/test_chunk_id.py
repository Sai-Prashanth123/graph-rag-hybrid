from config import RAGConfig
from document_processor import DocumentProcessor


def test_chunk_id_is_set_on_each_chunk():
    config = RAGConfig()
    processor = DocumentProcessor(config)

    text = "This is a test document. " * 200
    metadata = {"name": "test.txt", "type": "txt"}

    chunks = processor.process_documents([text], [metadata])

    assert len(chunks) >= 2, "expected multiple chunks for this length"
    for i, chunk in enumerate(chunks):
        expected_id = f"test.txt::{i}"
        assert chunk.metadata["chunk_id"] == expected_id, (
            f"chunk {i} has wrong chunk_id: {chunk.metadata.get('chunk_id')}"
        )


def test_chunk_id_uses_default_name_when_metadata_missing():
    config = RAGConfig()
    processor = DocumentProcessor(config)

    text = "Hello world. " * 200
    chunks = processor.process_documents([text])

    assert chunks[0].metadata["chunk_id"].startswith("document_0::")
