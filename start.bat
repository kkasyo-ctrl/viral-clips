@echo off
rem Double-click to start the Viral Clips web page; it opens in your browser.
rem Close this window to stop the server.
title Viral Clips
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.12 from python.org and try again.
    pause
    exit /b 1
)

python -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo Installing the web page's dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing dependencies failed; see the messages above.
        pause
        exit /b 1
    )
)

python webapp.py
if errorlevel 1 pause
