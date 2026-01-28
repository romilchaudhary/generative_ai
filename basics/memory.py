from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory # wraps a runnable with chat history, Loads old messages, Injects them into prompt
from langchain_core.chat_history import InMemoryChatMessageHistory # to store chat history, In-memory history
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="phi3:mini", temperature=0)

store = {}
def get_history(key: str) -> InMemoryChatMessageHistory:
    if key not in store:
        store[key] = InMemoryChatMessageHistory()
    return store[key]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
memory_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "user-1"}}
print(memory_chain.invoke({"input": "Hello, my name is Romil"}, config=config))
print(memory_chain.invoke({"input": "what is my name?"}, config=config))