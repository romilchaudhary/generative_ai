from langchain_community.chat_models import ChatOllama
from langchain_community.tools import Tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic.schema import Document

from dotenv import load_dotenv

load_dotenv()
"""
User Question
   ↓
Agent (phi3:mini)
   ↓ decides
Tool → Vector DB (Chroma)
   ↓ returns docs
Agent uses docs
   ↓
Final Answer
"""

# create vector db andtools using a vector store
llm = ChatOllama(model="phi3:mini", temperature=0, num_ctx=1024)
embeddings = OllamaEmbeddings(model="phi3:mini")
documents=[
        Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "langchain"}),
        Document(page_content="Ollama provides local LLMs that can be run on your machine.", metadata={"source": "ollama"}),
    ]
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

def vectorstore_tool(query: str) -> str:
    """Tool that queries the vector store and returns relevant documents."""
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

vector_tool = Tool(
    name="VectorStoreSearch",
    func=vectorstore_tool,
    description="A tool that searches a vector store for relevant documents to answer user queries.",
)
# create agent(LLM that chooses tools)
# - An agent is an LLM that: Reads the user question, Decides which tool
# - Calls the tool, Uses the result to answer
# Think: "LLM + reasoning + tool calling"
agent = initialize_agent(
    tools=[vector_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.invoke("What is LangChain?")
print(response)

"""
This agent works like this:
AgentType.ZERO_SHOT_REACT_DESCRIPTION means: Think → Use Tool → Observe → Think → Use Tool → Observe → ...
The agent will keep looping until it decides:
“I have enough information to answer.”
It retries when:
The tool output is too short
The tool output is ambiguous
The model is uncertain
The question feels open-ended
So the model thinks:
“Maybe I should search again to be sure.”
sol:1 Return complete, confident text from the tool.
    return f"""
    SOURCE: Internal KB
    ANSWER:
    {docs[0].page_content}
    """
sol:2 Add a stopping instruction
    agent = initialize_agent(
        tools=[db_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={
            "system_message": (
                "You must use tools at most once. "
                "If you have information, answer immediately."
            )
        }
    )
sol3: Limit agent iterations
    agent = initialize_agent(
        tools=[db_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=2
    )
sol4: Use a Single-Tool Agent (recommended for RAG)
agent = initialize_agent(
    tools=[db_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # or TOOL_CALLING in newer versions, this call the tool at most once
    verbose=True
)
"""