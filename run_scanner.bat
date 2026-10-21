@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Malicious PDF Detector - Interactive Scanner
echo ============================================================
echo.

:: Check for .venv
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Running setup first...
    call setup.bat
    if not exist ".venv\Scripts\activate.bat" (
        echo [ERROR] Setup failed or was cancelled. Cannot start scanner.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

:: Check for model bundle
if not exist "models\deployment\deployment_bundle_v1.joblib" (
    echo.
    echo [INFO] Deployment model bundle not found.
    echo        Building validated local bundle from verified mirror...
    python scripts\build_validated_bundle.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to build deployment bundle.
        pause
        exit /b 1
    )
    echo [INFO] Bundle built successfully.
)

echo.
echo [INFO] Starting Streamlit Scanner Application...
echo        Open your browser at http://localhost:8501 if it does not open automatically.
echo.
streamlit run app/streamlit_app.py
pause
