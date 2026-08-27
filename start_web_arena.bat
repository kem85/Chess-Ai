@echo off
chcp 65001 >nul
title ♟️ Chess-AI: Web Arena Launcher
cls

echo =====================================================================
echo   ♟️  CHESS-AI: DEEP RESIDUAL NEURAL ENGINE ^& WEB ARENA
echo   ⚡  Powered by ONNX Runtime INT8 Quantization ^& AlphaZero MCTS
echo =====================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10+ was not found in your PATH!
    echo Please install Python from https://www.python.org/downloads/ and ensure
    echo "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

echo [*] Launching Chess-AI Web Arena on http://127.0.0.1:8000 ...
echo [*] Press Ctrl+C in this terminal window to stop the server anytime.
echo.

python app.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Server exited with an error code. Checking dependencies...
    echo [*] Attempting to install required packages from requirements.txt...
    python -m pip install -r requirements.txt
    echo.
    echo [*] Retrying Web Arena launch...
    python app.py
)

pause
