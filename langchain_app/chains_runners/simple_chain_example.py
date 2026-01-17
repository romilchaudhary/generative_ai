from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="phi3", temperature=0.0)
prompt = PromptTemplate(
    input_variables=["question"],
    template="give answer in one line with some additional information of answer\nQuestion: {question}\nAnswer:"
)
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    while True:
        user_input = input("Enter your question: ")
        if user_input.strip() == "exit":
            print("bye")
            exit(0)
        if user_input.strip() == "":
            print("Please enter a valid question.")
            continue
        result = chain.invoke({"question": user_input})
        print(result)