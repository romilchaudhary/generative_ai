# generative_ai — examples

This repository contains small example apps that demonstrate local model wiring and Streamlit UIs.

Folders of interest

- `langchain_app/` — Streamlit app showing how to call a local Ollama model via the LangChain Ollama wrapper (`lang_2.py`).
- `streamlit_app/` — Simple two-column Streamlit demo that generates sample datasets using NumPy and pandas (`stream.py`).

Each subfolder contains a README with setup and run instructions. Below are quick start notes for Windows (PowerShell).

## Quick start (PowerShell)

1) Create and activate a Python virtual environment for the examples

```powershell
python -m venv .venv
# PowerShell activation
.\.venv\Scripts\Activate.ps1
```

2) Per-app installation and run

- LangChain + Ollama example (`langchain_app`):

  - Install dependencies

    ```powershell
    pip install -r .\langchain_app\requirements.txt
    ```

  - Ensure Ollama is installed and the daemon is running. If you use the `phi3` model (default in the app), pull it locally:

    ```powershell
    ollama pull phi3
    ```

  - Run the Streamlit app

    ```powershell
    streamlit run .\langchain_app\lang_2.py
    ```

  - Open the Streamlit URL (usually http://localhost:8501) and use the sidebar to set `Ollama base URL` (default `http://localhost:11434`) and model name, then press Connect / Reload Ollama.

- Streamlit demo (`streamlit_app`):

  - Install dependencies

    ```powershell
    pip install -r .\streamlit_app\requirements.txt
    ```

  - Run the app

    ```powershell
    streamlit run .\streamlit_app\stream.py
    ```

## Troubleshooting & notes

- If PowerShell blocks activating the virtual environment, you can run the batch activate script from cmd.exe: `\.venv\Scripts\activate`.
- The LangChain Ollama wrapper requires the Ollama daemon and the chosen model available locally; if you hit 404 or "model not found" when calling the app, run `ollama pull <model>` and reconnect using the app sidebar.
- Consider creating per-app virtual environments if you want to isolate dependencies.

## Need help?

If you want, I can:

- run the Streamlit apps here to verify they launch, or
- add small helper scripts (PowerShell) to create the venv and install deps for each app.

---

Requirements covered:

- Replace top-level README with consolidated instructions — Done