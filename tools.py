from langchain_core.tools import tool
import wikipedia
import json

@tool
def wiki_tool(topic: str) -> str:
    "Search Wikipedia for a topic and return summary"
    return wikipedia.summary(topic, sentences=3)

@tool
def save_tool(data: str, filename: str = "research.json") -> str:
    "Save a summary to a local file"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return f"Saved to {filename}"