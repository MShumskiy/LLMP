# LLMP Startup Script
# This script starts Ollama server and LLMP FastAPI application

Write-Host "🚀 Starting LLMP (Local LLM Platform)..." -ForegroundColor Green
Write-Host "=" * 50

# Check if Ollama is installed
try {
    $null = Get-Command ollama -ErrorAction Stop
    Write-Host "✅ Ollama found" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama not found. Please install Ollama first." -ForegroundColor Red
    Write-Host "   Download from: https://ollama.ai/" -ForegroundColor Yellow
    exit 1
}

# Check if Poetry is installed
try {
    $null = Get-Command poetry -ErrorAction Stop
    Write-Host "✅ Poetry found" -ForegroundColor Green
} catch {
    Write-Host "❌ Poetry not found. Please install Poetry first." -ForegroundColor Red
    Write-Host "   Install with: pip install poetry" -ForegroundColor Yellow
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating template..." -ForegroundColor Yellow
    @"
# Ollama Configuration
url=http://localhost:11434/api/

# Authentication
authentication_key=your_secret_key_here
ALLOWED_IPS=127.0.0.1,192.168.1.0/24

# Database Configuration
db_user=your_db_user
db_password=your_db_password
db_host=localhost
db_port=5432
db_database=llmp_db

# Client Configuration (for utils)
LLMP_URL=http://localhost:8000/generate
LLMP_PASSWORD=your_secret_key_here
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "📝 Please edit .env file with your configuration" -ForegroundColor Yellow
}

Write-Host "`n🔧 Starting services..." -ForegroundColor Cyan

# Function to cleanup on exit
function Cleanup {
    Write-Host "`n🛑 Shutting down services..." -ForegroundColor Yellow
    if ($ollamaJob) {
        Stop-Job $ollamaJob -ErrorAction SilentlyContinue
        Remove-Job $ollamaJob -ErrorAction SilentlyContinue
    }
    if ($llmpJob) {
        Stop-Job $llmpJob -ErrorAction SilentlyContinue
        Remove-Job $llmpJob -ErrorAction SilentlyContinue
    }
    Write-Host "✅ Cleanup completed" -ForegroundColor Green
}

# Register cleanup on Ctrl+C
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup }

try {
    # Start Ollama server in background
    Write-Host "🦙 Starting Ollama server..." -ForegroundColor Blue
    $ollamaJob = Start-Job -ScriptBlock { ollama serve } -Name "OllamaServer"
    
    # Wait a bit for Ollama to start
    Write-Host "⏳ Waiting for Ollama to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    # Check if Ollama is responding
    $maxRetries = 10
    $retryCount = 0
    $ollamaReady = $false
    
    while ($retryCount -lt $maxRetries -and -not $ollamaReady) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $ollamaReady = $true
                Write-Host "✅ Ollama server is ready!" -ForegroundColor Green
            }
        } catch {
            $retryCount++
            Write-Host "⏳ Waiting for Ollama... ($retryCount/$maxRetries)" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    
    if (-not $ollamaReady) {
        Write-Host "❌ Ollama server failed to start properly" -ForegroundColor Red
        exit 1
    }
    
    # Start LLMP FastAPI server
    Write-Host "🌐 Starting LLMP FastAPI server..." -ForegroundColor Blue
    Write-Host "📍 Server will be available at: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📖 API docs at: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop both services" -ForegroundColor Magenta
    Write-Host "=" * 50
    
    # Run LLMP in foreground so we can see logs and stop with Ctrl+C
    poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    
} catch {
    Write-Host "❌ Error occurred: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Cleanup
}
