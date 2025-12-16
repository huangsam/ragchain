"""Small demo runner that waits for services and exercises the API.

It expects CHROMA_SERVER_URL to be set (or will use in-process Chroma) and
that the ragchain API is reachable at http://ragchain:8000 when run via docker-compose.
"""

import asyncio
import os
import time

import aiohttp

API_URL = os.environ.get("RAGCHAIN_API_URL", "http://ragchain:8000")
HEALTH = f"{API_URL}/health"
INGEST = f"{API_URL}/ingest"
SEARCH = f"{API_URL}/search"


async def wait_for(url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(url, timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def post_json(session: aiohttp.ClientSession, url: str, payload: dict, timeout: int = 30):
    try:
        async with session.post(url, json=payload, timeout=timeout) as r:
            text = await r.text()
            return r.status, text
    except Exception as exc:
        return None, str(exc)


async def main():
    print("Waiting for ragchain API to be healthy...", flush=True)
    ok = await wait_for(HEALTH, timeout=60)
    if not ok:
        print("ragchain API didn't become healthy in time", flush=True)
        raise SystemExit(2)

    print("Triggering ingest for TIOBE top languages (sample list)", flush=True)
    titles = [
        "Python_(programming_language)",
        "C_(programming_language)",
        "C%2B%2B",
        "Java_(programming_language)",
        "C_Sharp_(programming_language)",
        "JavaScript",
        "SQL",
        "PHP",
        "R_(programming_language)",
        "Go_(programming_language)",
        "Swift_(programming_language)",
        "MATLAB",
        "Ruby_(programming_language)",
        "Visual_Basic",
        "Assembly_language",
        "Objective-C",
        "Fortran",
        "Kotlin_(programming_language)",
        "Dart_(programming_language)",
        "TypeScript",
    ]

    async with aiohttp.ClientSession() as session:
        status, text = await post_json(session, INGEST, {"titles": titles}, timeout=900)
        print("Ingest response:", status, text, flush=True)

        # Run a few sample searches to verify ingestion
        for q in ["python", "java", "javascript"]:
            print(f"Searching for '{q}'", flush=True)
            status, text = await post_json(session, SEARCH, {"query": q, "n_results": 1}, timeout=30)
            print("Search response:", status, text, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
