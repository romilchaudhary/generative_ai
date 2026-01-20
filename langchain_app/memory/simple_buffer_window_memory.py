from langchain_community.llms.ollama import Ollama
from langchain_classic.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

llm = Ollama(
    model="phi3",
    temperature=0.7,
)
# What it stores
"""
✔ Only last N messages
"""
# Best for:
"""
Save tokens
Short context
"""
# custom memory for conversation history
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=2  # retain last 2 exchanges
)
template = """You are a helpful assistant, give answer in one sentence. The conversation so far:
{chat_history}
User: {input}
Assistant:
"""
prompt = PromptTemplate(input_variables=["chat_history", "input"], template=template)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

if __name__ == "__main__":
    while True:
        input_question = input("Enter your question: ")
        if input_question.strip().lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        if not input_question.strip():
            print("Please enter a valid question.")
            continue
        if input_question:
            response = chain.invoke(input_question)
            print("ConversationChain Response:", response)
