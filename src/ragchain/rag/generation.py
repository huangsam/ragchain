import os
from typing import List

import ollama

# Default to qwen3 (usually 8b) for a good balance of speed and quality.
# We'll default to "qwen3" and let the user override via env var.
DEFAULT_MODEL = "qwen3"


class OllamaGenerator:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        # If running in Docker, we might need to point to host.docker.internal
        self.base_url = base_url or os.environ.get("OLLAMA_HOST")

        if self.base_url:
            self.client = ollama.AsyncClient(host=self.base_url)
        else:
            self.client = ollama.AsyncClient()

    async def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer using Ollama based on the provided context chunks.
        """
        context_text = "\n\n".join(context_chunks)

        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question:
{query}
"""

        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return response["message"]["content"]
