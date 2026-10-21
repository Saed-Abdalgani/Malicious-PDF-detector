@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Malicious PDF Detector - Environment Setup
echo ============================================================
echo.

:: Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking Python version...
python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Python version might be below 3.10. Recommended: Python 3.11.
) else (
    echo       Python version OK.
)

:: Create Virtual Environment
if not exist ".venv" (
    echo.
    echo [2/3] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully.
) else (
    echo.
    echo [2/3] Virtual environment (.venv) already exists.
)

:: Install Dependencies
echo.
echo [3/3] Installing / updating required dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo   To launch the PDF Scanner application:
echo       run_scanner.bat
echo.
echo   To run test verification:
echo       run_tests.bat
echo ============================================================
pause
