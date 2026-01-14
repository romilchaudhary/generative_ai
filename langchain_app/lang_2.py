import os
import streamlit as st

# Use the LangChain Ollama wrapper
# Install: pip install streamlit langchain-ollama
# Also make sure Ollama daemon is running and the phi3 model is available locally.
from langchain_community.llms import Ollama

st.set_page_config(page_title="LangChain + Ollama (phi3)", layout="wide")
st.title("LangChain + Ollama — phi3 Chat Example")

# -- Simple UI for model configuration --
with st.sidebar:
    st.header("Model settings")
    base_url = st.text_input(
        "Ollama base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        placeholder="http://localhost:11434",
    )
    model_name = st.text_input("Ollama model name", value="phi3")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=32, max_value=4096, value=1024, step=32)
    load = st.button("Connect / Reload Ollama")

# -- Session state initialization --
if "llm" not in st.session_state:
    st.session_state.llm = None
if "history" not in st.session_state:
    st.session_state.history = []

# -- Connect to Ollama when requested --
if load:
    if not base_url or not model_name:
        st.sidebar.error("Please provide Ollama base URL and model name before connecting.")
    else:
        st.sidebar.info("Connecting to Ollama (this may take a moment)...")
        try:
            # Instantiate Ollama LLM via LangChain wrapper
            llm = Ollama(model=model_name, base_url=base_url, temperature=float(temperature))
            # Optionally call a test to ensure connectivity (comment out if not desired)
            # _ = llm("Hello")
            st.session_state.llm = llm
            st.sidebar.success(f"Ollama model '{model_name}' connected.")
        except Exception as e:
            st.session_state.llm = None
            st.sidebar.error(f"Failed to connect to Ollama: {e}")

# -- Chat interface --
st.subheader("Chat")
if st.session_state.llm is None:
    st.warning("No Ollama model connected. Use the sidebar to set base URL, model name (e.g., phi3) and connect.")
    st.stop()

# Input form to submit user message
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Your message", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    # Build a simple conversational prompt that includes prior turns
    history_text = ""
    for turn in st.session_state.history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt = f"You are a helpful assistant.\n\n{history_text}User: {user_input}\nAssistant:"

    try:
        # Call the Ollama model via LangChain wrapper.
        # The Ollama wrapper may not be directly callable; prefer .generate (LangChain LLM API),
        # or fall back to .complete or calling if available.
        llm = st.session_state.llm
        reply = None

        if hasattr(llm, "generate"):
            # LangChain style: generate accepts a list of prompts and returns an LLMResult
            result = llm.generate([prompt])
            # Extract text from LLMResult.generations if present
            gens = getattr(result, "generations", None)
            if gens and len(gens) > 0 and len(gens[0]) > 0:
                reply = gens[0][0].text
            else:
                reply = str(result)
        elif hasattr(llm, "complete"):
            # Some wrappers expose a 'complete' method
            result = llm.complete(prompt)
            if isinstance(result, dict):
                reply = result.get("content") or result.get("text") or str(result)
            else:
                reply = str(result)
        elif callable(llm):
            # Last resort: if the object is callable, call it
            result = llm(prompt)
            if isinstance(result, dict):
                reply = result.get("content") or result.get("text") or str(result)
            else:
                reply = str(result)
        else:
            raise RuntimeError("Ollama object is not callable and has no supported interface (generate/complete).")
    except Exception as e:
        # Provide a clearer message when the model is not found (HTTP 404) and suggest how to fix it.
        err_str = str(e)
        if "404" in err_str or "not found" in err_str.lower() or "model not found" in err_str.lower():
            reply = (
                f"[Error calling Ollama model: {e}] "
                f"Possible cause: model '{model_name}' not found on Ollama at {base_url}. "
                f"Try running: `ollama pull {model_name}` locally, then press Connect / Reload Ollama in the sidebar."
            )
        else:
            reply = f"[Error calling Ollama model: {e}]"

    # Save to history
    st.session_state.history.append({"user": user_input, "assistant": reply})

# Display chat history
for turn in st.session_state.history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")
    st.write("---")

st.info("Using Ollama phi3. Next: add system prompts, streaming, or RAG as needed.")
