from langchain_community.llms.ollama import Ollama
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_classic.prompts import PromptTemplate

llm = Ollama(
    model="phi3",
    temperature=0.7,
)
# Memory stores past conversation data
# It is automatically added to the prompt
# Helps the bot remember context
# What it stores
"""
✔ Full conversation
✔ No limits
"""
# Best for:
"""
Learning
Small chats
"""
# use a custom key for storing conversation history
memory = ConversationBufferMemory(memory_key="custom_history")

template = """You are a helpful assistant, give answer in one sentence. The conversation so far:
{custom_history}
User: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["custom_history", "input"], template=template)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
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
