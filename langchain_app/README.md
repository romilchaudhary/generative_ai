# LangChain + Ollama (phi3) — Setup & run

This directory contains `lang_2.py`, a small Streamlit app that demonstrates calling a local Ollama model (default: `phi3`) through the LangChain Ollama wrapper.

This README explains how to prepare the Python environment, ensure Ollama is available, and run the app on Windows (PowerShell).

## Requirements

- Python 3.8+ (3.10/3.11 recommended)
- Ollama installed and the daemon running locally with the model you want to use (example: `phi3`).
- A Python virtual environment and the Python dependencies below.

Files in this folder:

- `lang_2.py` — Streamlit app.
- `requirements.txt` — minimal Python dependencies required for the app.

## Quick setup (PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
# Activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1
```

> If ExecutionPolicy blocks running the PowerShell activate script, you can instead run the activate batch file from cmd.exe:
>
> ```powershell
> .\.venv\Scripts\activate
> ```

2) Install Python dependencies

```powershell
pip install -r .\langchain_app\requirements.txt
```

3) Ensure Ollama daemon + model

- Make sure the Ollama daemon is running on your machine and the model (default `phi3`) is available locally. If the model is missing, pull it locally:

```powershell
ollama pull phi3
```

- Start the Ollama daemon per your Ollama installation instructions. The Streamlit app defaults to `http://localhost:11434` as `OLLAMA_BASE_URL`.

4) (Optional) Set `OLLAMA_BASE_URL` environment variable for the session

```powershell
# Example (PowerShell):
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

5) Run the Streamlit app

```powershell
streamlit run .\langchain_app\lang_2.py
```

Open the URL printed by Streamlit (usually http://localhost:8501) and use the sidebar to configure `Ollama base URL`, `Ollama model name` (default `phi3`), and press Connect / Reload Ollama.

## Environment & configuration notes

- The app reads `OLLAMA_BASE_URL` from the environment by default. You can also edit the value in the Streamlit sidebar.
- The code attempts to use LangChain's LLM API (`generate`) and falls back to other interfaces if needed. If you see an error that mentions `404` or "model not found", run `ollama pull <model_name>` and reconnect from the sidebar.

## Troubleshooting

- If connection to Ollama fails, check that the daemon is running and that `base_url` is correct.
- If Streamlit doesn't start because PowerShell execution policy prevents activation, either use the activate batch script in `\.venv\Scripts\activate` (cmd.exe) or temporarily adjust the policy per your security rules.

## Next steps / improvements

- Add system prompts, streaming, or RAG flows.
- Add a `docker-compose` or a script that starts Ollama and the Streamlit app together (if you use a reproducible local dev environment).

---

Requirements coverage:

- Create a README describing how to setup and run `lang_2.py` — Done
