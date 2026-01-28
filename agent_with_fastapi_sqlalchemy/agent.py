from langchain_community.llms.ollama import Ollama
from langchain_classic.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import calculator, company_info
from db_tools import get_user_by_email_tool

from langchain_classic import hub

model = Ollama(model="phi3", temperature=0)
tools = [calculator, company_info, get_user_by_email_tool]


prompt = hub.pull("hwchase17/react")


# - For production:

# - ❌ Avoid classic initialize_agent

# ✅ Use LangGraph

# ReAct pattern - LLM reasons + calls tools + itrerates + final answer
# agent = initialize_agent(
#     tools = tools,
#     llm = model,
#     max_iterations = 3,
#     agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose = True,
#     agent_kwargs={
#         "prefix": """
#         After using required tools ONCE,
#         produce a FINAL ANSWER and stop.
#         Do not repeat tool calls.
#         """
#     }
# )


# tool calling agent
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

