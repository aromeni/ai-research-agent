import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import wiki_tool, save_tool

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research assistant. Answer like this:\n{format_instructions}"),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [wiki_tool, save_tool]

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What would you like me to research? ")

try:
    raw_response = agent_executor.invoke({"query": query})
    structured_response = parser.parse(raw_response)
    print(structured_response)
except Exception as e:
    print("❌ Error:", e)
# Save the response to a file
if structured_response:
    save_tool(structured_response.json(), "research.json")
    print("✅ Research saved to research.json")
