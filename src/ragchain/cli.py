"""CLI for ragchain."""

import asyncio
import logging
import os

import click
import httpx

from ragchain.config import config
from ragchain.loaders import load_tiobe_languages, load_wikipedia_pages
from ragchain.rag import ingest_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@click.group()
def cli():
    """RAG pipeline CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(host, port):
    """Start the FastAPI server for RAG endpoints.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
    """
    import uvicorn

    uvicorn.run("ragchain.api:app", host=host, port=port, reload=True)


@cli.command()
@click.option("--n", default=10, help="Number of languages to ingest")
def ingest(n):
    """Ingest programming language documents into local vector store.

    Fetches top-n from TIOBE, loads Wikipedia articles, splits them,
    and stores in Chroma for semantic search.

    Args:
        n: Number of languages to ingest (default: 10)
    """

    async def _ingest():
        click.echo(f"Fetching top {n} languages from TIOBE...")
        langs = await load_tiobe_languages(n)
        click.echo(f"Fetched {len(langs)} languages: {', '.join(langs)}")

        click.echo("Loading Wikipedia pages...")
        docs = await load_wikipedia_pages(langs)
        click.echo(f"Loaded {len(docs)} documents from languages: {', '.join({d.metadata.get('language', 'Unknown') for d in docs})}")

        click.echo("Ingesting into vector store...")
        result = await ingest_documents(docs)
        click.echo(f"Result: {result}")

    asyncio.run(_ingest())


@cli.command()
@click.argument("query")
@click.option("--k", default=4, help="Number of results")
def search(query, k):
    """Search ingested documents using semantic similarity.

    Args:
        query: Search query (positional argument)
        k: Number of results to return (default: 4)
    """

    async def _search():
        from ragchain.rag import search as search_func

        result = await search_func(query, k=k)
        click.echo(f"Query: {result['query']}")
        for i, res in enumerate(result["results"], 1):
            metadata = res.get("metadata", {})
            title = metadata.get("title", "Unknown") if isinstance(metadata, dict) else "Unknown"
            click.echo(f"\n{i}. {title}")
            click.echo(f"   {res['content'][:200]}...")

    asyncio.run(_search())


@cli.command()
@click.argument("query")
@click.option("--model", default=config.ollama_model)
def ask(query, model):
    """Ask a question and get an answer using RAG + LLM.

    Sends question to the API server for retrieval and LLM-based generation.

    Args:
        query: Question to ask (positional argument)
        model: LLM model to use for generation (default: config.ollama_model)
    """

    async def _ask():
        # Increase timeout since LLM generation can take 30-60 seconds
        async with httpx.AsyncClient(timeout=120.0) as client:
            click.echo("Asking question (this may take a while for LLM generation)...")
            resp = await client.post(f"{config.ragchain_api_url}/ask", json={"query": query, "model": model})
            resp.raise_for_status()
            result = resp.json()
            click.echo(f"\nQ: {result['query']}")
            click.echo(f"A: {result['answer']}")

    asyncio.run(_ask())


if __name__ == "__main__":
    cli()
