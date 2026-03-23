@echo off
echo ========================================
echo   BookNView ML Service Startup
echo ========================================
echo.

:: Change to the ml-service directory
cd /d "%~dp0"

:: Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [!] No virtual environment found - using system Python
)

echo.
echo [*] Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo [*] Checking if model exists...
if not exist "model\demand_model.pkl" (
    echo [!] Model not found. Training model first...
    python train_model.py
    if errorlevel 1 (
        echo [ERROR] Model training failed. Please check your setup.
        pause
        exit /b 1
    )
)

echo.
echo [*] Starting FastAPI ML Service on port 8085...
echo [*] API docs:   http://localhost:8085/docs
echo [*] Health:     http://localhost:8085/health
echo [*] Prediction: POST http://localhost:8085/predict-demand
echo.
uvicorn predict_api:app --reload --port 8085 --host 127.0.0.1

pause
