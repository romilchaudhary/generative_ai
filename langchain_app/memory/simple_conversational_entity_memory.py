from langchain_community.llms.ollama import Ollama
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationEntityMemory
from langchain_core.prompts import PromptTemplate

llm = Ollama(
    model="phi3",
    temperature=0.7,
)
# remember facts
# Facts about people, places, things
# Best for:
# - Remembering key details
# - Quick retrieval of information
# - User profiles
# - personal assistant applications

# memory
memory = ConversationEntityMemory(
    llm=llm,
    max_token_limit=1024
)

# template
# ConversationEntityMemory provides 'history' and 'entities' keys, so the prompt must accept them.
template = """You are a helpful assistant, give answer in one sentence. The conversation so far:
{history}
Known entities: {entities}
User: {input}
Assistant:
"""

prompt = PromptTemplate(
    input_variables=["history", "entities", "input"],
    template=template
)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

chain.invoke(input="My name is romil")
print(chain.invoke(input="What is my name?"))