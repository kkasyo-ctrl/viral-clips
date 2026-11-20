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

python -c "import flask, yt_dlp" >nul 2>nul
if errorlevel 1 (
    echo Installing the web page's dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing dependencies failed; see the messages above.
        pause
        exit /b 1
    )
)

rem Sites change often and old yt-dlp versions stop working, so keep it current.
rem Offline, this fails quietly and the installed version is used.
echo Checking for yt-dlp updates...
python -m pip install --quiet --upgrade --disable-pip-version-check "yt-dlp[default]" >nul 2>nul

where deno >nul 2>nul
if errorlevel 1 echo Note: Deno is not installed, so YouTube links may fail. See the README.

python webapp.py
if errorlevel 1 pause
