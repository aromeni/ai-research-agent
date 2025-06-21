from __future__ import annotations

from langchain_core.tools import tool
import wikipedia
import json
from typing import Optional


# ------------------------------
# 1.  Wikipedia lookup tool
# ------------------------------
@tool
def wiki_tool(topic: str) -> str:
    """
    Return a short summary (first paragraph) *plus the page URL*.

    Fallback strategy:
    1.  Exact title – “Quantum computing”
    2.  Underscore variant – “Quantum_computing”
    3.  First search hit from `wikipedia.search()`
    If nothing works, return a clear 'Page not found …' string.
    """

    def try_page(title: str) -> Optional[str]:
        """Return 'summary • url' line for *title*, or None if not found."""
        try:
            page = wikipedia.page(title, auto_suggest=False)
            first_paragraph = page.summary.split("\n")[0]
            return f"{first_paragraph}\nSOURCE: {page.url}"
        except (wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError):
            return None
        except Exception as exc:
            return f"Unexpected Wikipedia error: {exc}"

    # 1) exact match
    result = try_page(topic)
    if result:
        return result

    # 2) underscores
    result = try_page(topic.replace(" ", "_"))
    if result:
        return result

    # 3) search fallback
    try:
        hits = wikipedia.search(topic, results=1)
        if hits:
            result = try_page(hits[0])
            if result:
                return result
    except Exception as exc:
        return f"Unexpected Wikipedia search error: {exc}"

    # Nothing worked
    return f"Page not found for topic: {topic}"


# ----------------------------------------------------------------------
# 2.  Save-to-disk tool
# ----------------------------------------------------------------------
@tool
def save_tool(data: str | dict, filename: str = "research.json") -> str:
    """
    Save *data* (string or dict) to *filename* in JSON format.
    Returns a status message instead of raising.
    """
    try:
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        return f"Saved to {filename}"
    except Exception as exc:
        return f"Failed to save: {exc}"
