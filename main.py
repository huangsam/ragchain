import argparse
import asyncio
import os
from pathlib import Path

from ragchain.rag.ingest import ingest
from ragchain.vectorstore.chroma_vectorstore import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description="ragchain CLI")
    parser.add_argument("--titles", type=str, help="Comma-separated Wikipedia titles to ingest")
    parser.add_argument("--save-dir", type=str, default="wikipages", help="Directory to save raw pages")
    args = parser.parse_args()

    titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    save_dir = Path(args.save_dir)

    # If a CHROMA_SERVER_URL or CHROMA_PERSIST_DIRECTORY is configured in the
    # environment, use a ChromaVectorStore so ingest will upsert vectors into
    # the running Chroma instance. This makes the newcomer flow seamless.
    server_url = os.environ.get("CHROMA_SERVER_URL")
    persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY")
    vectorstore = None
    if server_url or persist_dir:
        vectorstore = ChromaVectorStore(server_url=server_url, persist_directory=persist_dir)

    report = asyncio.run(ingest(titles, save_dir=save_dir, vectorstore=vectorstore))
    print("Ingest report:", report)


if __name__ == "__main__":
    main()
