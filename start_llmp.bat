@echo off
title LLMP Startup Script
color 0A

echo.
echo ===============================================
echo    🚀 Starting LLMP (Local LLM Platform)
echo ===============================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Ollama not found. Please install Ollama first.
    echo    Download from: https://ollama.ai/
    pause
    exit /b 1
)
echo ✅ Ollama found

REM Check if Poetry is installed
where poetry >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Poetry not found. Please install Poetry first.
    echo    Install with: pip install poetry
    pause
    exit /b 1
)
echo ✅ Poetry found

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Creating template...
    echo # Ollama Configuration > .env
    echo url=http://localhost:11434/api/ >> .env
    echo. >> .env
    echo # Authentication >> .env
    echo authentication_key=your_secret_key_here >> .env
    echo ALLOWED_IPS=127.0.0.1,192.168.1.0/24 >> .env
    echo. >> .env
    echo # Database Configuration >> .env
    echo db_user=your_db_user >> .env
    echo db_password=your_db_password >> .env
    echo db_host=localhost >> .env
    echo db_port=5432 >> .env
    echo db_database=llmp_db >> .env
    echo. >> .env
    echo # Client Configuration ^(for utils^) >> .env
    echo LLMP_URL=http://localhost:8000/generate >> .env
    echo LLMP_PASSWORD=your_secret_key_here >> .env
    echo.
    echo 📝 Please edit .env file with your configuration
    echo.
)

echo.
echo 🔧 Starting services...
echo.

REM Start Ollama server in background
echo 🦙 Starting Ollama server...
start "Ollama Server" /MIN cmd /c "ollama serve"

REM Wait for Ollama to start
echo ⏳ Waiting for Ollama to initialize...
timeout /t 5 /nobreak > nul

echo ✅ Ollama server started!
echo.

REM Start LLMP FastAPI server
echo 🌐 Starting LLMP FastAPI server...
echo 📍 Server will be available at: http://localhost:8000
echo 📖 API docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the LLMP server
echo (Ollama will continue running in background)
echo ===============================================
echo.

poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --reload

echo.
echo 🛑 LLMP server stopped.
pause
