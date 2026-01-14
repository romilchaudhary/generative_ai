# Streamlit Example App — Setup & Run

This folder contains `stream.py`, a small two-column Streamlit demo that generates and displays sample datasets (Sine/Cosine or Random) using NumPy and pandas.

This README explains how to prepare a Python environment on Windows (PowerShell), install dependencies, and run the app.

## Files

- `stream.py` — the Streamlit app.

## Requirements

- Python 3.8+
- Virtual environment (recommended)
- Packages: `streamlit`, `pandas`, `numpy`

## Quick setup (PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
# PowerShell activate
.\.venv\Scripts\Activate.ps1
# If ExecutionPolicy blocks Activate.ps1, run the batch activate from cmd.exe instead:
# .\.venv\Scripts\activate
```

2) Install dependencies

```powershell
pip install streamlit pandas numpy
```

3) Run the app

```powershell
# from the repository root (examples folder)
streamlit run .\streamlit_app\stream.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## Notes & Troubleshooting

- The app includes a short "Setup & Run" code snippet in the sidebar as a quick reference.
- If Streamlit is not found after installing, ensure the activated virtual environment is in use (check `python -V` and `which streamlit` or `Get-Command streamlit`).
- If you need to install into a specific Python interpreter, use its pip (`.venv\Scripts\pip install ...`).

## Optional improvements

- Add a `requirements.txt` for reproducible installs.
- Provide a PowerShell script to create and activate the venv and install deps.

---

Requirement covered:

- Create a README describing how to setup and run `stream.py` — Done
