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





from langchain.tools import tool
import wikipedia
from duckduckgo_search import DDGS
import requests, feedparser, json
from typing import List


# --- existing wiki_tool & save_tool stay as-is ---------------------------


@tool
def ddg_search_tool(query: str, max_results: int = 5) -> List[str]:
    """DuckDuckGo web search – returns a list of result URLs."""
    with DDGS() as ddg:
        results = (
            r["href"]
            for r in ddg.text(query, safesearch="moderate", max_results=max_results)
        )
    return list(results)


@tool
def arxiv_tool(topic: str, max_results: int = 3) -> List[str]:
    """Return arXiv paper titles + links for *topic*."""
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{topic.replace(' ', '+')}&start=0&max_results={max_results}"
    )
    feed = feedparser.parse(requests.get(url, timeout=10).text)
    papers = [
        f"{entry.title.strip()} – {entry.link}"
        for entry in feed.entries[:max_results]
    ]
    return papers


@tool
def news_tool(query: str, max_results: int = 5) -> List[str]:
    """Very simple news RSS search via DuckDuckGo."""
    rss = f"https://duckduckgo.com/rss/news?q={query.replace(' ', '+')}"
    items = feedparser.parse(rss).entries[:max_results]
    return [f"{it.title} – {it.link}" for it in items]



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
