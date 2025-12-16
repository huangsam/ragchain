import os
from pathlib import Path

import pytest

from ragchain.vectorstore.chroma_vectorstore import ChromaVectorStore


@pytest.fixture(params=["inprocess", "remote"])
def chroma_store(request, tmp_path: Path):
    """Yield a ChromaVectorStore in either in-process (persistent) or remote HTTP mode.

    - If 'remote' is requested but CHROMA_SERVER_URL is not set, the test is skipped.
    - Each store uses a unique collection name derived from the test name to avoid collisions.
    """
    # Skip entire module if chromadb isn't available
    try:
        import chromadb  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"chromadb not installed or unusable: {exc}")

    mode = request.param
    # sanitize name to conform to Chroma collection naming rules
    raw_name = f"test_{request.node.name}_{os.getpid()}"
    import re

    name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)
    # ensure starts and ends with alphanumeric
    if not re.match(r"[A-Za-z0-9]", name[0]):
        name = f"c{name}"
    if not re.match(r"[A-Za-z0-9]", name[-1]):
        name = f"{name}0"

    if mode == "remote":
        server_url = os.environ.get("CHROMA_SERVER_URL")

        def _http_ok(url, timeout=1):
            import urllib.request

            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    return 200 <= resp.status < 400
            except Exception:
                return False

        if not server_url:
            # If a local server is available at 127.0.0.1:8000, use it. Otherwise skip and instruct the user.
            # Prefer v2 heartbeat; fall back to legacy /health and root / if needed.
            if (
                _http_ok("http://127.0.0.1:8000/api/v2/heartbeat")
                or _http_ok("http://127.0.0.1:8000/v2/health")
                or _http_ok("http://127.0.0.1:8000/health")
                or _http_ok("http://127.0.0.1:8000/")
            ):
                server_url = "http://127.0.0.1:8000"
            else:
                pytest.skip(
                    "CHROMA_SERVER_URL not set and no local Chroma detected at http://127.0.0.1:8000; "
                    "run `ragchain up --profile test` or set CHROMA_SERVER_URL to run remote chroma tests",
                )
            # propagate for any code that reads it
            os.environ["CHROMA_SERVER_URL"] = server_url

        store = ChromaVectorStore(server_url=server_url, collection_name=name)
        yield store
    else:
        persist_dir = tmp_path / f"chroma_{request.node.name}"
        store = ChromaVectorStore(persist_directory=str(persist_dir), collection_name=name)
        yield store
