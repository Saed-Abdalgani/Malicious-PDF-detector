@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Malicious PDF Detector - Test Suite
echo ============================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Running setup first...
    call setup.bat
    if not exist ".venv\Scripts\activate.bat" (
        echo [ERROR] Setup failed or was cancelled. Cannot run tests.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo [INFO] Running pytest test suite...
pytest -v
pause
