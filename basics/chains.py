from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from narwhals import Unknown
from narwhals import Unknown

# Chain = a pipeline of steps where the output of one step becomes the input of the next.
# “A chain is a deterministic pipeline that connects prompts, LLMs, tools, and parsers where each step’s output feeds into the next.”
# Input → Step 1 → Step 2 → Step 3 → Final Output
# Each step depends on the previous one → that’s a chain.

# Rule of thumb:
    # - Known steps → Chain
    # - Unknown steps / tool decisions → Agent

# types - LLM Chains, Sequential Chains, RAG Chains, Tool chain etc.
# runnable chains using | operator

# invoke() - returns full response
# - invoke_as_string() - returns string output only
# - invoke_as_dict() - returns output as dict
# stream() - for real-time token generation
# - stream_as_string() - streams string output
# batch() - for processing multiple inputs
# - batch_as_string() - returns list of string outputs


llm = Ollama(model="phi3", temperature=0)
prompt = PromptTemplate(
    input_variables=["product"],
    template="Give me a creative name for a company that makes {product}."
)

chain = prompt | llm | StrOutputParser()

response = chain.stream({"product": "iphone"})
print(response)