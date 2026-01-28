from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "first_project")

llm = ChatOllama(model="phi3:mini", temperature=0, num_ctx=1024)

# create an in-memory chat history
class SimpleChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []

# session based history
session_store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = SimpleChatMessageHistory()
    return session_store[session_id]

# create prompt
# MessagesPlaceholder is mandatory for memory.
prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# create chain
chain = prompt | llm

# create runnable with message history
chat_memory_runnable = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    ai_message_cls=AIMessage,
    human_message_cls=HumanMessage,
    input_messages_key="input",
    history_messages_key="history"
)

if __name__ == "__main__":
    session_id = "user_123"
    user_inputs = [
        "Hello! Who won the world series in 2020?",
        "Where was it played?",
        "Who was the MVP?"
    ]

    for user_input in user_inputs:
        response = chat_memory_runnable.invoke(
            {"input": user_input}, config={"configurable": {"session_id": session_id}}
        )
        print(f"User: {user_input}")
        print(f"AI: {response}\n")