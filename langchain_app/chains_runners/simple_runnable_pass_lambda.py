from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# RunnablePassthrough
# - Takes input and passes it forward unchanged
# - Keep original input for downstream processing
# - Useful for debugging and logging
# - Forward user question to multiple steps

passthrough = RunnablePassthrough()


# RunnableLambda
# - Custom processing with a Python function
# - Allows for more complex logic and transformations
# - Format text , Clean data, Extract fields
# - Combine context

uppercase = RunnableLambda(lambda x: x.upper())
def custom_function_call(text: str) -> str:
    return f"Custom Function Processed: {text[::-1]}"  # reverses the text and adds a prefix
custom_function = RunnableLambda(custom_function_call)

# chaining
chained_result = (passthrough | uppercase | custom_function)
input_text = 'what is the capital of france?'

# invoke each runnable separately to see intermediate outputs
# passthrough_out = passthrough.invoke(input_text)
# uppercase_out = uppercase.invoke(passthrough_out)
# custom_function_out = custom_function.invoke(uppercase_out)

# print("passthrough output:", passthrough_out)
# print("uppercase output:", uppercase_out)
# print("custom function output:", custom_function_out)

# final chained invocation (should match uppercase_out)
final_out = chained_result.invoke(input_text)
print("chained_result output:", final_out)
