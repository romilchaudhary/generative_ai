from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOllama as Ollama
from langchain_community.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType

load_dotenv()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "first_project")

llm = Ollama(model="phi3:mini",temperature=0, num_ctx=1024)

# create tool
# - A tool is just a Python function that the LLM can call.
# - Example: Calculator, Current time, Custom Python logic
def get_weather_tool(location: str) -> str:
    """Tool that returns a fake weather report for a given location."""
    return f"The weather in {location} is sunny with a high of 75°F."
weather_tool = Tool(
    name="GetWeather",
    func=get_weather_tool,
    description="A tool that provides the weather for a given location.",
)

def calculator_tool(query: str) -> str:
    """Tool that uses a calculator to answer questions."""
    return {eval(query)}

cal_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="A calculator tool that evaluates mathematical expressions.",
)

# create agent(LLM that chooses tools)
# - An agent is an LLM that: Reads the user question, Decides which tool to use, 
# - Calls the tool, Uses the result to answer

# Think: "LLM + reasoning + tool calling"

agent = initialize_agent(
    tools=[cal_tool, weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
# What this agent does:
# - Uses ReAct reasoning
# - Reads tool descriptions
# - Decides tool usage automatically

# When should you use agents?
# - ✅ Multi-step task
# - ✅ Decision-making

response = agent.invoke("What is 12 * 5 + 3 ?")
print(response)
response = agent.invoke("Explain fastapi in one sentence and then calculate 12345 * 6789 ?")
print(response)
