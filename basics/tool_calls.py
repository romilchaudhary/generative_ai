from langchain_community.llms.ollama import Ollama
from langchain_classic.tools import tool
from langchain_classic.agents import initialize_agent, AgentType

model = Ollama(model="phi3", temperature=0)

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))


@tool
def company_info(name: str) -> str:
    """Get company info"""
    data = {
        "tata": "Tata Group is an Indian multinational conglomerate.",
        "google": "Google is a global technology company."
    }
    return data.get(name.lower(), "Company not found")

tool = [calculator, company_info]
agent = initialize_agent(
    tools = tool,
    llm = model,
    max_iterations = 1,
    agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)
agent.invoke("what is the output of 2+2-1?")
agent.invoke("give me a brief info about tata?")

