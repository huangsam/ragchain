# Skip if chromadb is missing
import importlib.util

import pytest
from aioresponses import aioresponses

from ragchain.rag.embeddings import DummyEmbedding
from ragchain.rag.ingest import ingest

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)


@pytest.mark.asyncio
async def test_full_ingest_pipeline(tmp_path, chroma_store):
    # 1. Setup mocks and components
    title = "Integration_Test_Page"
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    sections_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"

    summary_json = {"title": title, "extract": "This is the summary paragraph."}
    sections_json = {
        "sections": [
            {"line": "Section 1", "text": "<p>This is the content of section 1.</p>"},
            {"line": "Section 2", "text": "<p>This is the content of section 2.</p>"},
        ]
    }

    # Use the chroma_store fixture which parametrizes in-process and remote server modes.
    store = chroma_store

    # Use dummy embeddings (deterministic)
    emb = DummyEmbedding(dim=16)

    # 2. Run Ingest
    with aioresponses() as m:
        m.get(summary_url, payload=summary_json)
        m.get(sections_url, payload=sections_json)

        report = await ingest(
            titles=[title],
            save_dir=tmp_path / "wikipages",
            chunk_size=50,  # small chunk size to force multiple chunks
            overlap=10,
            embedding_client=emb,
            vectorstore=store,
        )

    # 3. Assertions
    assert report.pages_processed == 1
    assert report.chunks_created > 0
    assert report.vectors_upserted == report.chunks_created

    # Verify raw file was saved
    raw_file = tmp_path / "wikipages" / f"{title}.json"
    assert raw_file.exists()

    # 4. Verify Retrieval
    # Search for something that should match the summary
    query_vec = (await emb.embed_texts(["summary paragraph"]))[0]
    results = await store.search(query_vec, n_results=1)

    assert results["ids"]
    assert len(results["ids"][0]) > 0

    # Check metadata
    metadatas = results["metadatas"][0]
    assert metadatas[0]["title"] == title
    assert metadatas[0]["source"] == "wikipedia"

    # Check document content
    documents = results["documents"][0]
    assert "summary" in documents[0] or "content" in documents[0]
