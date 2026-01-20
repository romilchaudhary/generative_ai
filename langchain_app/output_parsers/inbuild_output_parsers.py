from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser
from langchain_classic.chains import LLMChain
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema # deprecated
from pydantic import BaseModel

# Output parser = converts LLM text into a usable format
# LLM text → Parser → Clean result
# StrOutputParser -> Returns the output as plain string. use cases: You just want clean text, Chatbots, Summaries etc.
# CommaSeparatedListOutputParser -> Returns the output as a list of strings, separated by commas. use cases: You want a list of items, Tags, Keywords etc.
# StructuredOutputParser -> Returns the output as a structured JSON object. use cases: You want to extract specific fields, API responses, etc.
# PydanticOutputParser -> Returns the output as a Pydantic model. use cases: You want to validate and parse the output into a specific schema.
# Which Parser Should You Use?
"""
    | Need          | Parser                           |
    | ------------- | -------------------------------- |
    | Plain text    | `StrOutputParser`                |
    | List          | `CommaSeparatedListOutputParser` |
    | JSON-like     | `StructuredOutputParser`         |
    | Strict schema | `PydanticOutputParser`           |
"""

# create llm model
llm = Ollama(
    model="phi3",
    temperature=0.7,
)
str_parser = StrOutputParser()
# print("str_parser created:", str_parser.get_format_instructions())
comma_separated_parser = CommaSeparatedListOutputParser()
print("comma_separated_parser created:", comma_separated_parser.get_format_instructions())
# create prompt template
str_output_prompt = PromptTemplate(
    input_variables=["question"],
    template="What is the answer to the following question? {question}"
)
comma_separated_prompt = PromptTemplate(
    input_variables=["question"],
    template="list the answers to the following question, separated by commas: {question}"
)
# structured output prompt
# schema = [
#     ResponseSchema(name="definition", description="What is SIP"),
#     ResponseSchema(name="benefit", description="One benefit of SIP")
# ]
# structured_output_parser = StructuredOutputParser.from_response_schemas(schema)
# print("structured_output_parser created:", structured_output_parser.get_format_instructions())
# structured_output_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="Provide a structured response to the following question: {question}\n{structured_output_parser.get_format_instructions()}"
# )
# create pydantic output parser
class PydanticResponseModel(BaseModel):
    definition: str
    benefit: str

pydantic_output_parser = PydanticOutputParser(pydantic_object=PydanticResponseModel)
print("pydantic_output_parser created:", pydantic_output_parser.get_format_instructions())
pydantic_output_prompt = PromptTemplate(
    input_variables=["question"],
    template="Provide a Pydantic response to the following question: {question}\n{format_instructions}",
    partial_variables={"format_instructions": pydantic_output_parser.get_format_instructions()}
)
# partial_variables - are values you fix in the prompt in advance
# Prompt = Fixed part (partial) + Dynamic part (user input)
# PydanticOutputParser requires format instructions to be included in the prompt.

# create stroutputparser llm chain
str_output_chain = LLMChain(
    llm=llm,
    prompt=str_output_prompt,
    output_parser=str_parser
)

# create comma separated list output parser llm chain
comma_separated_list_chain = LLMChain(
    llm=llm,
    prompt=comma_separated_prompt,
    output_parser=comma_separated_parser
)

# create structured output parser llm chain
# structured_output_chain = LLMChain(
#     llm=llm,
#     prompt=structured_output_prompt,
#     output_parser=structured_output_parser
# )

# create pydantic output parser llm chain
pydantic_output_chain = LLMChain(
    llm=llm,
    prompt=pydantic_output_prompt,
    output_parser=pydantic_output_parser
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
            response = str_output_chain.run(input_question)
            print("StrOutputParser Response:", response)
            response = comma_separated_list_chain.run(input_question)
            print("CommaSeparatedListOutputParser Response:", response)
            # response = structured_output_chain.run(input_question)
            # print("StructuredOutputParser Response:", response)
            # The chain is already configured with a Pydantic output parser, so run()
            # will return the parsed/validated result; avoid parsing it again.
            try:
                response = pydantic_output_chain.run(input_question)
                print("PydanticOutputParser Response:", response)
            except Exception as e:
                # Fallback: run the prompt without the chain's output_parser to get raw text,
                # then try to parse it manually with the PydanticOutputParser.
                print("Pydantic parser failed with error:", e)
                raw_chain = LLMChain(
                    llm=llm,
                    prompt=pydantic_output_prompt,
                )
                raw_output = raw_chain.run(input_question)
                print("Raw model output (unparsed):", raw_output)
                try:
                    parsed = pydantic_output_parser.parse(raw_output)
                    print("Parsed PydanticOutputParser Response (after manual parse):", parsed)
                except Exception as e2:
                    print("Failed to parse model output:", e2)