# lang.py
# Lightweight LangChain app using a small free HF model (distilgpt2).
# Requirements: pip install langchain transformers torch
# Run: python lang.py --prompt "Write a short poem about Python."

import argparse
import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

def create_hf_llm(model_name: str = "distilgpt2", max_new_tokens: int = 1000):
    """
    Create a small local Hugging Face text-generation pipeline and wrap it for LangChain.
    Uses GPU if available, otherwise CPU.
    """
    device = 0 if torch.cuda.is_available() else -1
    text_gen = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=text_gen)

def build_chain(llm):
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="You are a helpful assistant.\n\n{user_input}",
    )
    return LLMChain(llm=llm, prompt=prompt)

def main():
    parser = argparse.ArgumentParser(description="Tiny LangChain app with a lightweight HF LLM (distilgpt2).")
    parser.add_argument("--model", "-m", default="distilgpt2", help="Hugging Face model id (default: distilgpt2)")
    parser.add_argument("--prompt", "-p", help="Prompt to send to the model")
    parser.add_argument("--max_tokens", "-n", type=int, default=100, help="Max new tokens to generate")
    args = parser.parse_args()

    llm = create_hf_llm(model_name=args.model, max_new_tokens=args.max_tokens)
    chain = build_chain(llm)

    if args.prompt:
        user_input = args.prompt
    else:
        user_input = input("Enter prompt: ")

    response = chain.run(user_input)
    print("\n=== Response ===\n")
    print(response.strip())

if __name__ == "__main__":
    main()
