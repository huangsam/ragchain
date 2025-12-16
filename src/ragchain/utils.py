import re


def safe_filename(title: str) -> str:
    """Return a filesystem-safe filename for a given title."""
    # replace non-word chars with underscore
    name = re.sub(r"[^\w\-\.]+", "_", title)
    return name.strip("_")
