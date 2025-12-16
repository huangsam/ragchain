from ragchain.parser.html_parser import extract_text_from_mobile_sections


MOBILE_SECTIONS_SAMPLE = {
    "sections": [
        {"line": "Intro", "text": "<p>First paragraph <sup>1</sup></p>"},
        {"line": "History", "text": "<p>History paragraph</p>"},
    ]
}


def test_extract_text_from_mobile_sections():
    text = extract_text_from_mobile_sections(MOBILE_SECTIONS_SAMPLE)
    assert "Intro" in text
    assert "First paragraph" in text
    assert "History" in text
    assert "History paragraph" in text
