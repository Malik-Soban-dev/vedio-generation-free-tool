@echo off
REM ---------------------------------------------------------------------------
REM WanGP Studio Dashboard - local launcher (Windows)
REM
REM Opens the dashboard in a REAL new browser window at http://localhost:7860
REM No E2B sandbox, no traffic-access-token, no 403.
REM ---------------------------------------------------------------------------
setlocal

set PORT=7860
set URL=http://localhost:%PORT%/
pushd "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python was not found on your PATH.
    echo Install Python 3.8+ from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
  )
  set PY=py
) else (
  set PY=python
)

echo ==============================================================
echo   WanGP v18.17 Studio - Interactive Dashboard Preview
echo ==============================================================
echo   Opening : %URL%
echo   Press Ctrl+C in this window to stop the server.
echo ==============================================================

start "" "%URL%"
%PY% preview_server.py

popd
endlocal
