@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Dashboard environment is missing. Running setup first...
  call setup_windows.bat
  if errorlevel 1 exit /b 1
)

set "MPLCONFIGDIR=%TEMP%\capstone-v8-dashboard-matplotlib"
set "TF_CPP_MIN_LOG_LEVEL=2"
echo Starting the Capstone II v8.1 Dashboard at http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
exit /b %errorlevel%
