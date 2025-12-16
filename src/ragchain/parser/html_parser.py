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
    for sec in sections:
        html = sec.get("text") or ""
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        # remove edit links and sup marks
        for sup in soup.select("sup, .mw-editsection"):
            sup.decompose()
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
