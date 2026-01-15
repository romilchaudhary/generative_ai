# LangChain + Ollama (phi3) — Setup & run

This directory contains two Streamlit examples that demonstrate calling a local Ollama model via LangChain wrappers, and a simple retrieval-augmented generation (RAG) demo using Chromadb (Chroma).

Apps in this folder:

- `lang_2.py` — Simple chat UI that calls an Ollama model (default: `phi3`) via the LangChain Ollama wrapper.
- `simple_lang_rag.py` — Ingests a paragraph, stores embeddings in a Chromadb-backed Chroma index, and answers questions using a RetrievalQA chain (uses `OllamaEmbeddings` by default).

This README explains how to prepare the Python environment, ensure Ollama and Chromadb are available, and run the apps on Windows (PowerShell).

## Requirements

- Python 3.8+ (3.10/3.11 recommended)
- Ollama installed and the daemon running locally with the model you want to use (example: `phi3`).
- A Python virtual environment and the Python dependencies listed in `requirements.txt` in this folder.

Files in this folder:

- `lang_2.py` — Streamlit chat app using the Ollama wrapper.
- `simple_lang_rag.py` — Small RAG demo that uses chromadb/Chroma for retrieval.
- `requirements.txt` — Python dependencies for the examples.

## Quick setup (PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
# PowerShell activation
.\.venv\Scripts\Activate.ps1
```

If ExecutionPolicy blocks running the PowerShell activate script, you can instead run the activate batch file from cmd.exe:

```powershell
.\.venv\Scripts\activate
```

2) Install Python dependencies

```powershell
pip install -r .\langchain_app\requirements.txt
```

3) Ensure Ollama daemon & model

- Make sure the Ollama daemon is running on your machine and the model (default `phi3`) is available locally. If the model is missing, pull it locally:

```powershell
ollama pull phi3
```

- Start the Ollama daemon per your Ollama installation instructions. The examples default to `http://localhost:11434` as `OLLAMA_BASE_URL` unless you set the environment variable or adjust the UI.

4) Run either example

- Chat example (`lang_2.py`):

```powershell
streamlit run .\langchain_app\lang_2.py
```

- RAG example (`simple_lang_rag.py`):

```powershell
streamlit run .\langchain_app\simple_lang_rag.py
```

## Notes & Troubleshooting

- The RAG example uses `chromadb` / `Chroma`. Different versions of `chromadb` and `langchain` wrappers may require different constructor parameters; the example includes fallbacks to improve compatibility across releases.
- If you see errors indicating the model is not found or a 404 when calling Ollama, run `ollama pull <model_name>` and ensure the daemon is running and reachable at the configured base URL.
- If pip cannot find `langchain_community`, `ollama`, or other imports, check PyPI or the project docs for the current package name — some community wrappers are released under different names or via GitHub.
- The default Chroma persistence directory is `./chromadb_store`; delete it to reset the index.

## Next steps / improvements

- Add system prompts, streaming, or RAG flows.
- Add a `docker-compose` or a script that starts Ollama and the Streamlit app together (useful for reproducible local dev environments).

---

Requirements coverage:

- Create a combined README describing how to setup and run `lang_2.py` and `simple_lang_rag.py` — Done
