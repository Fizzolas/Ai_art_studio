@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV=%SCRIPT_DIR%venv"
set "LOG_DIR=%SCRIPT_DIR%logs"
set "LOG_FILE=%LOG_DIR%\app.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: ── CUDA memory-fragmentation prevention ──────────────────────────────
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6

:: ── Safeguard: tell HF hub to use the standard cache location ─────────
if not defined HF_HOME set HF_HOME=%USERPROFILE%\.cache\huggingface

:: Read version
set /p VERSION=<"%SCRIPT_DIR%VERSION"
echo [AI Art Studio v%VERSION%] Starting...

:: Check venv
if not exist "%VENV%\Scripts\activate.bat" (
    echo [WARN] Virtual environment not found.
    echo Run: python setup.py
    pause
    exit /b 1
)

:: Activate and run
call "%VENV%\Scripts\activate.bat"
python "%SCRIPT_DIR%main.py" >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application crashed. Last log entries:
    echo ----------------------------------------
    powershell -Command "Get-Content '%LOG_FILE%' -Tail 20"
    echo ----------------------------------------
    echo Full log: %LOG_FILE%
    pause
)
