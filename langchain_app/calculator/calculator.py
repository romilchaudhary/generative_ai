import re
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, LLMMathChain



"""
Calculator app using Ollama (via LangChain) to convert natural-language
calculation requests into a simple arithmetic expression (no math module or Python
builtins allowed) and use eval to compute the result.

@Developed By: Romil
"""

# -------------------------
# Helpers to interact with LLM
# -------------------------
def extract_expression(text: str) -> str:
    """
    Try to extract a concise expression from LLM output.
    Prefer code fences if present; otherwise take last non-empty line.
    Accept dict-like outputs from LLM chains by extracting common string fields.
    """
    # Normalize non-string LLM outputs (some chain APIs return dicts)
    if isinstance(text, dict):
        # common keys where text might be stored
        for key in ("text", "response", "output", "content", "answer", "result"):
            if key in text and isinstance(text[key], str):
                text = text[key]
                break
        else:
            # fallback: pick the first string value in the dict
            for v in text.values():
                if isinstance(v, str):
                    text = v
                    break
            else:
                # final fallback: stringify the whole object
                text = str(text)
    elif not isinstance(text, str):
        text = str(text)

    # code fence extraction
    fence_match = re.search(r"```(?:python)?\n([\s\S]+?)\n```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # single backtick extraction
    backtick_match = re.search(r"`([^`]+)`", text)
    if backtick_match:
        return backtick_match.group(1).strip()

    # otherwise take last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip()


# -------------------------
# Build LangChain Ollama LLM and prompts
# -------------------------
def build_llm(model: str = "phi3", temperature: float = 0.0):
    """
    Create an Ollama LLM instance via LangChain.
    The model name depends on your Ollama local/models setup.
    """
    return Ollama(model=model, temperature=temperature)


CONVERT_PROMPT = PromptTemplate(
    input_variables=["instruction"],
    template=(
        "You are a translator: convert the user's natural-language calculation request "
        "into a single-line arithmetic expression that uses only numeric literals, "
        "parentheses, and the operators +, -, *, /, **, //, and %. Do NOT use the Python "
        "math module (e.g., no math.sin) and do NOT use any other Python builtins (abs, round, etc.). "
        "If the user mentions functions like sin, cos, sqrt, pi, etc., compute their numeric values "
        "and substitute them into the expression (so the returned expression contains only numbers "
        "and arithmetic operators). Return only the expression (one line) and do not add explanation.\n\n"
        "Example input: '(3 + 5) * 2 + 5 squared + 1 + 2' -> output: (3 + 5) * 2 + 5**2 + 1 + 2\n\n"
        "User request:\n{instruction}\n"
    ),
)

EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["expr", "result"],
    template=(
        "Explain step-by-step how to compute the following arithmetic expression and the final result.\n\n"
        "Expression: {expr}\nResult: {result}\nProvide a concise, clear explanation (few steps)."
    ),
)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="LangChain + Ollama Calculator", layout="wide")
st.title("Calculator (LangChain + Ollama)")

with st.sidebar:
    st.header("Model settings")
    model_name = st.text_input(
        "Ollama model name",
        value="phi3",
        help="Name of the model available to your Ollama server/local installation",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    use_llm_for_explanation = st.checkbox(
        "Ask LLM for explanation", value=False, help="Get a short LLM explanation of the steps"
    )
    st.markdown("Make sure your Ollama daemon is running locally and models exist.")

st.markdown("Enter a calculation in natural language or as a simple arithmetic expression.")
user_input = st.text_area(
    "Calculation",
    value="(3 + 5) * 2 - sqrt(16)  # note: converter will replace functions with numbers",
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Compute"):
        if not user_input.strip():
            st.warning("Please enter a calculation.")
        else:
            llm = build_llm(model=model_name, temperature=temperature)
            convert_chain = LLMChain(llm=llm, prompt=CONVERT_PROMPT)

            try:
                raw = convert_chain.invoke(user_input)
            except Exception as e:
                st.error(f"LLM conversion failed: {e}")
                raw = ""

            expr = extract_expression(raw) if raw else user_input.strip()
            # remove only the first double-quote character, if present
            expr = expr.replace('"', '', 1)
            st.subheader("Extracted expression (only numbers & operators allowed)")
            st.code(expr, language="python")
            # llmmathchain is not supported in ollama phi3 or others
            # math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
            try:
                result = eval(expr)
                st.subheader("Result")
                st.success(result)
            except Exception as e:
                st.error(f"Math chain evaluation failed: {e}")
                result = None

            # Ask LLM to explain steps if requested and if evaluation succeeded
            if use_llm_for_explanation and result is not None:
                explain_chain = LLMChain(llm=llm, prompt=EXPLAIN_PROMPT)
                try:
                    explanation = explain_chain.run(expr=expr, result=result)
                    st.subheader("Explanation")
                    st.write(explanation)
                except Exception as e:
                    st.warning(f"LLM explanation failed: {e}")
