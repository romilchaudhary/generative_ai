from langchain_community.llms.ollama import Ollama
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

llm = Ollama(model="phi3", temperature=0)
prompt = PromptTemplate(
    input_variables=["input"],
    template="you are an ai assistant, give answer in one sentence {input}."
)

# Runnable = anything that can be executed with .invoke()
# A Runnable is a standard interface for components that take input and produce output. ex. prompts, LLMs, Chains, Tools, Agents, OutputParser, retrievers etc.
# Chains and Agents are also Runnables as they implement the .invoke() method.

# Input → Runnable → Output

# RunnablePassThrough - passes input as output without any changes
# RunnableLambda - wrap any function as a Runnable
# RunnableParrallel - run multiple Runnables in parallel
# RunnableSequence - run multiple Runnables in sequence

# 1. RunnablePassThrough
passthrough = RunnablePassthrough()
response = passthrough.invoke("Hello, World!")
print(response)  # Output: Hello, World!

# 2. RunnableLambda - turns a normal function into a Runnable.
def add_exclamation(text: str) -> str:
    return text + "!"

lambda_runnable = RunnableLambda(add_exclamation)
response = lambda_runnable.invoke("Hello, World")
print(response)  # Output: Hello, World!

lambda_runnable_async = RunnableLambda(lambda text: text.upper())
response = lambda_runnable_async.invoke("Hello, World")
print(response)  # Output: HELLO, WORLD!

# 3. RunnableParallel- (Run Things Together)
runnable1 = RunnableLambda(lambda text: text + " from Runnable 1")
runnable2 = RunnableLambda(lambda text: text + " from Runnable 2")
parallel = RunnableParallel(
    runnable1=runnable1,
    runnable2=runnable2
)
response = parallel.invoke("Hello, World")
print(response)  # Output: {'runnable1': 'Hello, World from Runnable 1', 'runnable2': 'Hello, World from Runnable 2'}

# 4. Runnable Sequence - (Run Things One After Another)
runnable_sequence = prompt | llm | StrOutputParser()
response = runnable_sequence.invoke({"input": "What is the capital of France?"})
print(response)  # Output: Paris

# You can also chain multiple Runnables together using the | operator.
complex_runnable = RunnableParallel(
    input=RunnablePassthrough()
) | prompt | llm | StrOutputParser()
response = complex_runnable.invoke({"input": "Give me a creative name for a company that makes smartphones."})
print(response)  # Output: TechNova

# runnable -single executable unit with .invoke() method
# chain - sequence of runnables connected together
# Chain is just a Runnable made of other Runnables