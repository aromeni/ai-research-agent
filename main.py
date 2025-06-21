import os
import json 
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ---- your custom tools -----------------------------------------------
from tools import (
    wiki_tool,
    ddg_search_tool,
    arxiv_tool,
    news_tool,
    save_tool,
)

# ----------------------------------------------------------------------
# 1. Load the OpenAI key
# ----------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------------
# 2. Structured-output schema  (tools_used made optional)
# ----------------------------------------------------------------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str] = []      # default → no validation crash


# Parser that turns raw LLM text into the schema above
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ----------------------------------------------------------------------
# 3. Language model
# ----------------------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

# ----------------------------------------------------------------------
# 4. Prompt template
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a research assistant.  Choose whichever of these tools helps:\n"
    "- wiki_tool        • encyclopedia summary\n"
    "- ddg_search_tool  • general web search\n"
    "- arxiv_tool       • academic papers\n"
    "- news_tool        • latest headlines\n"
    "- save_tool        • store the final JSON\n"
    "Return the FINAL answer strictly in the JSON format below.  "
    "If nothing is found, set summary='Not found', sources=[], tools_used=[].\n"
    "{format_instructions}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ----------------------------------------------------------------------
# 5. Register every tool once
# ----------------------------------------------------------------------
tools = [wiki_tool, ddg_search_tool, arxiv_tool, news_tool, save_tool]

# ----------------------------------------------------------------------
# 6. Build the tool-calling agent
# ----------------------------------------------------------------------
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------------------------------------------------
# 7. Run a query
# ----------------------------------------------------------------------
if __name__ == "__main__":
    query = input("What would you like me to research? ")

    try:
        raw = agent_executor.invoke({"query": query})

        # LangChain may return {"output": "..."} or plain str
        output_text = raw.get("output") if isinstance(raw, dict) else str(raw)

        result = parser.parse(output_text)
        print("\n✅ Final Structured Output:\n")
        # print(result.json(indent=2, ensure_ascii=False))
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


    except Exception as exc:
        print("❌ Error:", exc)
        print("Raw response:\n", raw)
