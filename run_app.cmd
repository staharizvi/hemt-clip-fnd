@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Install requirements.txt into .venv first.
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py %*
