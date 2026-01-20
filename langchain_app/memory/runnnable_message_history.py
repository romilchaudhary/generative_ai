from langchain_community.chat_models import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory


llm = ChatOllama(
    model="phi3",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful finance assistant.give answer in one line."),
    ("human", "{input}")
])
print(prompt.input_variables)
chain = prompt | llm

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_message_key="input"
)

if __name__ == "__main__":
    session_id = "user_session_1"
    while True:
        input_text = input("Enter your message (or 'exit' to quit): ")
        if input_text.strip().lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        if not input_text.strip():
            print("Please enter a valid message.")
            continue
        response = chatbot.invoke({"input": input_text}, config={"configurable": {"session_id": session_id}})
        print(response)