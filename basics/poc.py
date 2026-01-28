from langchain_community.chat_models import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOllama(model="phi3:mini", temperature=0.7)
# system_message = SystemMessage("You are a helpful assistant.")
# human_message = HumanMessage("Hello, how are you?")
# ai_message = AIMessage("I'm good.")

# messages = [system_message, human_message, ai_message]
# ai_message = model.invoke(messages)
# print(ai_message)

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")
human_msg = HumanMessage("How do I create a REST API?")
print(model.invoke([system_msg, human_msg]))