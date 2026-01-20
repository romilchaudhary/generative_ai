from langchain_community.llms.ollama import Ollama
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate

llm = Ollama(
    model="phi3",
    temperature=0.7,
)

# memory with custom key
memory = ConversationSummaryBufferMemory(
    memory_key="chat_history",
    llm=llm,
    max_token_limit=1024
)

#What it stores
"""
✔ Summary instead of full chat
✔ Uses LLM to compress memory
"""
# Best for:
"""
Long conversations
Token efficiency
"""

# prompt
template = """You are a helpful assistant, create summary of maximum 100 words. The conversation so far:
{chat_history}
User: {input}
Assistant:
"""
prompt = PromptTemplate(input_variables=["chat_history", "input"], template=template)

# chain
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
