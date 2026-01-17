from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Note: All parallel chains get the same input and Output is always a dictionary
# Both chains run at the same time.
# when to use parallel chains?
# - When you have multiple independent tasks that can be executed simultaneously
# - To improve response time by utilizing concurrency
# - When you want to gather different perspectives or types of information in one go
# examples
# - Getting a concise answer and a detailed explanation for a complex question
# - Simultaneously retrieving facts and opinions on a controversial topic
# - Comparing different approaches to solving a problem
# - Getting a quick summary and a detailed analysis at the same time

llm = Ollama(model="phi3", temperature=0.0)
# Chain 1: Short answer
short_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer in ONE sentence: {question}"
)
short_chain = short_prompt | llm | StrOutputParser()

# Chain 2: Simple explanation
explanation_prompt = PromptTemplate(
    input_variables=["question"],
    template="Explain in simple terms and provide a brief explanation: {question}"
)
explanation_chain = explanation_prompt | llm | StrOutputParser()

# Run both chains in parallel
parallel_chain = RunnableParallel(
    short_answer=short_chain,
    long_answer=explanation_chain
)

if __name__ == "__main__":
    while True:
        user_input = input("Enter your question: ")
        if user_input.strip() == "exit":
            print("bye")
            exit(0)
        if user_input.strip() == "":
            print("Please enter a valid question.")
            continue
        # Invoke
        result = parallel_chain.invoke({
            "question": user_input
        })
        print(result)
        print("\n=== Short Answer ===\n")
        print(result["short_answer"])
        print("\n=== Long Answer ===\n")
        print(result["long_answer"])