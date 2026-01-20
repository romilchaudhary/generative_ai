import os
import sys
from langchain_community.llms import Ollama

def get_chat_response(prompt: str, model: str = "phi3") -> str:
    llm = Ollama(model=model)
    result = llm.invoke(prompt)
    return result

if __name__ == "__main__":
    prompt = "Hello! Give a short example response from the API."
    try:
        answer = get_chat_response(prompt)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
