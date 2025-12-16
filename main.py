import argparse
import asyncio
from pathlib import Path

from ragchain.rag.ingest import ingest


def main():
    parser = argparse.ArgumentParser(description="ragchain CLI")
    parser.add_argument("--titles", type=str, help="Comma-separated Wikipedia titles to ingest")
    parser.add_argument("--save-dir", type=str, default="wikipages", help="Directory to save raw pages")
    args = parser.parse_args()

    titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    save_dir = Path(args.save_dir)

    report = asyncio.run(ingest(titles, save_dir=save_dir))
    print("Ingest report:", report)


if __name__ == "__main__":
    main()
