import pytest

from ragchain.rag.embeddings import DummyEmbedding
from ragchain.vectorstore.chroma_vectorstore import ChromaVectorStore


@pytest.mark.asyncio
async def test_chroma_upsert_and_search(tmp_path):
    client = None
    # create store with persist dir to isolate
    store = ChromaVectorStore(client=client, collection_name="test_collection", persist_directory=str(tmp_path))
    emb = DummyEmbedding(dim=16)

    texts = ["hello world", "goodbye world"]
    vectors = await emb.embed_texts(texts)

    docs = [
        {"id": "d1", "text": texts[0], "metadata": {"title": "hello"}, "embedding": vectors[0]},
        {"id": "d2", "text": texts[1], "metadata": {"title": "goodbye"}, "embedding": vectors[1]},
    ]

    await store.upsert_documents(docs)

    query_vec = (await emb.embed_texts(["hello world"]))[0]
    res = await store.search(query_vec, n_results=1)

    # chroma returns list-of-lists for ids and distances
    ids = res.get("ids")
    assert ids and isinstance(ids, list)
    # the first result's first id should be d1
    first_id = ids[0][0] if isinstance(ids[0], list) else ids[0]
    assert first_id == "d1"
