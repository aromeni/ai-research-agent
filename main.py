import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import wiki_tool, save_tool

# Load OpenAI key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define structured output model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Configure model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt with agent_scratchpad and format instructions
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful research assistant. Use the available tools.\n"
         "Always return the final answer strictly in the JSON format below.\n"
         "If you cannot find information, set the 'summary' field to "
         "'Not found', leave 'sources' empty, but STILL return valid JSON.\n"
         "{format_instructions}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())



# Define tools
tools = [wiki_tool, save_tool]

# Create agent
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Prompt user
query = input("What would you like me to research? ")

# Execute and parse
structured_response = None
try:
    raw_response = agent_executor.invoke({"query": query})

    # Ensure clean parsing
    if isinstance(raw_response, dict) and "output" in raw_response:
        output_text = raw_response["output"]
    else:
        output_text = str(raw_response)

    structured_response = parser.parse(output_text)
    print("\n✅ Final Structured Output:\n")
    print(structured_response)

except Exception as e:
    print("❌ Error:", e)
    print("Raw response:\n", raw_response)
