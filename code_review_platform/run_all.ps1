# run_all.ps1 - Convenience script to start CodeSentinel AI backend (FastAPI) and UI (Streamlit)
# -----------------------------------------------------------------------------------------
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition

# 1. Kill any existing processes using ports 8000 (FastAPI) or 8501 (Streamlit)
Write-Host "Checking for existing processes on ports 8000 and 8501..." -ForegroundColor Cyan

$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($port8000) {
    foreach ($conn in $port8000) {
        Write-Host "Stopping process ($($conn.OwningProcess)) using port 8000..." -ForegroundColor Yellow
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

$port8501 = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($port8501) {
    foreach ($conn in $port8501) {
        Write-Host "Stopping process ($($conn.OwningProcess)) using port 8501..." -ForegroundColor Yellow
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# 2. Check virtual environment path (located in parent folder of code_review_platform)
$ParentPath = Split-Path -Parent -Path $PSScriptRoot
$pythonExe = Join-Path $ParentPath ".venv\Scripts\python.exe"
$streamlitExe = Join-Path $ParentPath ".venv\Scripts\streamlit.exe"

if (!(Test-Path $pythonExe) -or !(Test-Path $streamlitExe)) {
    Write-Error "Virtual environment not properly initialized. Make sure `.venv` folder exists inside $ParentPath."
    exit 1
}

# 3. Start FastAPI Backend in background
Write-Host "Starting FastAPI Backend..." -ForegroundColor Cyan
$uvicornArgs = @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload")
$backendProc = Start-Process -FilePath $pythonExe -ArgumentList $uvicornArgs -NoNewWindow -PassThru
Write-Host "FastAPI Backend started successfully." -ForegroundColor Green

# Give FastAPI a moment to start
Start-Sleep -Seconds 2

# 4. Start Streamlit UI in background
Write-Host "Starting Streamlit UI..." -ForegroundColor Cyan
$streamlitArgs = @("run", "streamlit_app.py", "--server.headless=true", "--browser.gatherUsageStats=false")
$uiProc = Start-Process -FilePath $streamlitExe -ArgumentList $streamlitArgs -NoNewWindow -PassThru
Write-Host "Streamlit UI started successfully." -ForegroundColor Green

Write-Host "`nAll services have been started!" -ForegroundColor Green
Write-Host "FastAPI: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "Streamlit UI: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to terminate both servers..." -ForegroundColor Yellow

# Wait for both processes to complete or for user interruption
try {
    while ($true) {
        if ($backendProc.HasExited) {
            Write-Warning "FastAPI backend has stopped."
            break
        }
        if ($uiProc.HasExited) {
            Write-Warning "Streamlit UI has stopped."
            break
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host "`nStopping servers..." -ForegroundColor Red
    if ($backendProc -and !$backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($uiProc -and !$uiProc.HasExited) {
        Stop-Process -Id $uiProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Cleaned up all processes." -ForegroundColor Green
}
