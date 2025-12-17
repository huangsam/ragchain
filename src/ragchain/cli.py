"""CLI for ragchain."""

import asyncio
import os

import click
import httpx

from ragchain.loaders import load_tiobe_languages, load_wikipedia_pages
from ragchain.rag import ingest_documents

API_URL = os.environ.get("RAGCHAIN_API_URL", "http://localhost:8000")


@click.group()
def cli():
    """RAG pipeline CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(host, port):
    """Start the API server."""
    import uvicorn

    uvicorn.run("ragchain.api:app", host=host, port=port, reload=True)


@cli.command()
@click.option("--n", default=10, help="Number of languages to ingest")
def ingest(n):
    """Ingest programming languages from TIOBE."""

    async def _ingest():
        click.echo(f"Fetching top {n} languages from TIOBE...")
        langs = await load_tiobe_languages(n)
        click.echo(f"Fetched {len(langs)} languages: {', '.join(langs)}")

        click.echo("Loading Wikipedia pages...")
        docs = await load_wikipedia_pages(langs)
        click.echo(f"Loaded {len(docs)} documents from languages: {', '.join(set(d.metadata.get('language', 'Unknown') for d in docs))}")

        click.echo("Ingesting into vector store...")
        result = await ingest_documents(docs)
        click.echo(f"Result: {result}")

    asyncio.run(_ingest())


@cli.command()
@click.argument("query")
@click.option("--k", default=4, help="Number of results")
def search(query, k):
    """Search the vector store."""

    async def _search():
        from ragchain.rag import search as search_func

        result = await search_func(query, k=k)
        click.echo(f"Query: {result['query']}")
        for i, res in enumerate(result["results"], 1):
            click.echo(f"\n{i}. {res['metadata'].get('title', 'Unknown')}")
            click.echo(f"   {res['content'][:200]}...")

    asyncio.run(_search())


@cli.command()
@click.argument("query")
@click.option("--model", default="qwen3")
def ask(query, model):
    """Ask a question using RAG + LLM."""

    async def _ask():
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_URL}/ask", json={"query": query, "model": model})
            resp.raise_for_status()
            result = resp.json()
            click.echo(f"Q: {result['query']}")
            click.echo(f"A: {result['answer']}")

    asyncio.run(_ask())


if __name__ == "__main__":
    cli()
