import os
from typing import List, Dict, Any

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

    async def generate(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using Ollama based on the provided context items.
        Each item should have 'text' and 'meta' keys.
        """
        # Format context with XML-like tags for better separation
        formatted_context = []
        for i, item in enumerate(context_items, 1):
            title = item.get("meta", {}).get("title", "Unknown Source")
            text = item.get("text", "").strip()
            formatted_context.append(f"<document index='{i}' title='{title}'>\n{text}\n</document>")

        context_block = "\n\n".join(formatted_context)

        system_prompt = (
            "You are a precise and helpful assistant. "
            "You must answer the user's question based ONLY on the provided context documents. "
            "Cite the document titles if relevant. "
            "If the answer is not in the context, say 'I don't have enough information to answer that.' "
            "Do not use outside knowledge."
        )

        user_prompt = f"""Context:
{context_block}

Question:
{query}
"""

        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.0,  # Deterministic output
            },
        )

        return response["message"]["content"]
