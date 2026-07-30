@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>nul
if errorlevel 1 goto try_python

py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=py -3.12"
  goto python_found
)

py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=py -3.11"
  goto python_found
)

:try_python
where python >nul 2>nul
if errorlevel 1 goto python_missing
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
if errorlevel 1 goto python_missing
set "PYTHON_CMD=python"

:python_found
if not exist ".venv\Scripts\python.exe" (
  echo Creating Dashboard environment with %PYTHON_CMD%...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 exit /b 1
)

echo Installing complete Dashboard and Sandbox dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements-dashboard.txt
if errorlevel 1 exit /b 1

set "TF_CPP_MIN_LOG_LEVEL=2"
echo Verifying Streamlit, LR/RF, XGBoost, LSTM, and Sandbox runtime...
".venv\Scripts\python.exe" -c "import streamlit, pandas, numpy, matplotlib, seaborn, sklearn, xgboost, tensorflow, joblib, exchange_calendars, pyarrow; print('Dashboard runtime verified.')"
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Run run_dashboard.bat.
exit /b 0

:python_missing
echo Python 3.11 or 3.12 was not found.
echo Install Python 3.12, enable Add Python to PATH, and run this file again.
exit /b 1
