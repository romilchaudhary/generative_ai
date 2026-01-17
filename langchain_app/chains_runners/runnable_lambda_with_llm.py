from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = Ollama(model="phi3", temperature=0)

prompt = PromptTemplate(
    template="give answer in one sentence: What is the capital of {country}?",
    input_variables=["country"]
)

passthrough_chain = RunnablePassthrough()
country_result_chain = prompt | llm | StrOutputParser()


parallel_country_chain = RunnableParallel(
    country = passthrough_chain,
    capital = country_result_chain
)
# get summary country
country_summary_chain = PromptTemplate(
    template="Provide a brief summary about the countrywith not more than 100 words: {capital}",
    input_variables=["capital"]
) | llm | StrOutputParser()

final_chain = parallel_country_chain | country_summary_chain
summary = final_chain.invoke("France")
print(summary)