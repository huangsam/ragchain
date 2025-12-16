from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from ragchain.parser.html_parser import (
    extract_first_paragraph_from_summary,
    extract_text_from_mobile_sections,
)
from ragchain.parser.wiki_client import fetch_wikipedia_pages
from ragchain.rag.chunker import chunk_text
from ragchain.rag.embeddings import DummyEmbedding, EmbeddingClient
from ragchain.utils import safe_filename


@dataclass
class IngestReport:
    pages_processed: int = 0
    pages_skipped: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    errors: List[str] = None


async def ingest(
    titles: Iterable[str],
    save_dir: Optional[Path] = None,
    chunk_size: int = 1000,
    overlap: int = 200,
    embedding_client: Optional[EmbeddingClient] = None,
    vectorstore: Optional[Any] = None,
) -> IngestReport:
    """High-level ingest orchestration: fetch, parse, chunk, embed, and optionally store.

    If vectorstore is provided, documents are upserted.
    """
    save_dir = Path(save_dir) if save_dir else None
    if embedding_client is None:
        embedding_client = DummyEmbedding()

    pages = await fetch_wikipedia_pages(list(titles), save_dir=save_dir)
    report = IngestReport(pages_processed=0, pages_skipped=0, chunks_created=0, vectors_upserted=0, errors=[])

    for p in pages:
        report.pages_processed += 1
        # prefer summary extract for quick preview
        summary = p.get("summary") or {}
        first_para = extract_first_paragraph_from_summary(summary) or ""
        sections = p.get("sections") or {}
        full_text = extract_text_from_mobile_sections(sections)
        text_to_chunk = first_para + "\n\n" + full_text if first_para else full_text
        
        chunks = chunk_text(text_to_chunk, chunk_size=chunk_size, overlap=overlap)
        report.chunks_created += len(chunks)
        
        embeddings = await embedding_client.embed_texts(chunks)
        
        if vectorstore:
            docs = []
            title_safe = safe_filename(p.get("title", "unknown"))
            for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
                doc_id = f"{title_safe}_{i}"
                docs.append({
                    "id": doc_id,
                    "text": chunk,
                    "metadata": {
                        "title": p.get("title", ""),
                        "chunk_index": i,
                        "source": "wikipedia"
                    },
                    "embedding": vec
                })
            await vectorstore.upsert_documents(docs)
            report.vectors_upserted += len(docs)

    return report
