@echo off
title AI Art Studio
echo Starting AI Art Studio...

:: ── CUDA memory-fragmentation prevention ──────────────────────────────
:: max_split_size_mb   = prevent the allocator from splitting blocks >128 MB
:: garbage_collection_threshold = run GC more aggressively (60 %% fill)
:: These significantly reduce "CUDA out of memory" errors on 8 GB GPUs.
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6

:: ── Safeguard: tell HF hub to use the standard cache location ─────────
if not defined HF_HOME set HF_HOME=%USERPROFILE%\.cache\huggingface

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe main.py
) else (
    echo Virtual environment not found. Running setup first...
    python setup.py
    echo.
    echo Setup complete. Starting application...
    venv\Scripts\python.exe main.py
)
pause
