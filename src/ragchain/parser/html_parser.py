from __future__ import annotations

from typing import Dict, List, Optional

from bs4 import BeautifulSoup


def extract_text_from_mobile_sections(sections_json: Dict) -> str:
    """Extract readable text from Wikipedia mobile-sections JSON.

    The mobile-sections API returns 'sections' where each section has an
    'text' field containing HTML for that section.
    """
    parts: List[str] = []
    sections = sections_json.get("sections") or []
    # Iterate through each section returned by the mobile-sections API.
    # Some sections may be empty; skip them early to avoid unnecessary work.
    for sec in sections:
        html = sec.get("text") or ""
        if not html:
            # Empty section (no HTML) — skip.
            continue
        soup = BeautifulSoup(html, "html.parser")
        # Remove edit links and superscript markers that aren't useful for
        # text search / summarization. These nodes may contain noisy UI
        # artifacts (e.g., '[edit]') which we want to exclude.
        for sup in soup.select("sup, .mw-editsection"):
            sup.decompose()
        # Extract normalized text with newlines separating blocks.
        text = soup.get_text(separator="\n").strip()
        if text:
            heading = sec.get("line")
            if heading:
                parts.append(heading)
            parts.append(text)
    return "\n\n".join(parts).strip()


def extract_first_paragraph_from_summary(summary_json: Dict) -> Optional[str]:
    """Return the extract (first paragraphs) from the summary endpoint if present."""
    return summary_json.get("extract")
