# AI Plagiarism Detection System - Sprint 2 Demo Script
param(
    [string]$Mode = "local",  # Options: local, api, docker
    [string]$ApiKey = $null   # Custom API key (generates secure one if not provided)
)

Write-Host "🚀 AI Plagiarism Detection System - Sprint 2" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Yellow

# Generate secure API key if not provided
if (-not $ApiKey) {
    $ApiKey = [System.Web.Security.Membership]::GeneratePassword(32, 8)
    Write-Host "🔐 Generated secure API key: $($ApiKey.Substring(0,8))..." -ForegroundColor Cyan
}

Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host ""

switch ($Mode.ToLower()) {
    "local" {
        Write-Host "🏠 Starting in Local Mode..." -ForegroundColor Green
        Write-Host "This mode uses embedded DocumentProcessor (no API calls)" -ForegroundColor Gray
        Write-Host ""
        
        # Create venv if it doesn't exist
        if (-not (Test-Path ".venv")) {
            Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
            python -m venv .venv
            if ($LASTEXITCODE -ne 0) {
                Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
                exit 1
            }
        }
        
        # Activate venv
        Write-Host "⚡ Activating virtual environment..." -ForegroundColor Yellow
        .\.venv\Scripts\Activate.ps1
        
        # Upgrade pip
        Write-Host "🔧 Upgrading pip..." -ForegroundColor Yellow
        python -m pip install --upgrade pip
        
        # Install deps with compatibility fixes
        Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
        pip install huggingface_hub==0.25.2
        pip install sentence-transformers==2.2.2
        pip install -r requirements.txt
        
        # Pre-warm model
        Write-Host "🤖 Pre-warming AI model (prevents demo delays)..." -ForegroundColor Yellow
        python -c "
from sentence_transformers import SentenceTransformer
print('🔥 Caching AI model for instant demo...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✅ AI model ready for flawless demonstration!')
"
        
        # Create data directory
        Write-Host "📁 Creating data directory..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path "data" -Force | Out-Null
        
        # Run tests
        Write-Host "🧪 Running system tests..." -ForegroundColor Yellow
        python -m pytest tests/ -q
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ All tests passed!" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Some tests failed, but continuing..." -ForegroundColor Yellow
        }
        
        # Start Streamlit in local mode
        Write-Host ""
        Write-Host "🎯 Starting Streamlit (Local Mode)..." -ForegroundColor Green
        Write-Host "🌐 Demo available at: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "✅ AI model pre-warmed - Zero loading delays during demo!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Press Ctrl+C to stop the demo" -ForegroundColor Gray
        
        streamlit run apps/frontend/demo_streamlit.py
    }
    
    "api" {
        Write-Host "🌐 Starting in API Mode..." -ForegroundColor Green
        Write-Host "This mode runs FastAPI backend + Streamlit frontend" -ForegroundColor Gray
        Write-Host ""
        
        # Setup venv
        if (-not (Test-Path ".venv")) {
            Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
            python -m venv .venv
        }
        
        Write-Host "⚡ Activating virtual environment..." -ForegroundColor Yellow
        .\.venv\Scripts\Activate.ps1
        
        Write-Host "🔧 Upgrading pip..." -ForegroundColor Yellow
        python -m pip install --upgrade pip
        
        Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
        pip install huggingface_hub==0.25.2
        pip install sentence-transformers==2.2.2  
        pip install -r requirements.txt
        
        # Pre-warm model
        Write-Host "🤖 Pre-warming AI model..." -ForegroundColor Yellow
        python -c "
from sentence_transformers import SentenceTransformer
print('🔥 Caching AI model for instant demo...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✅ AI model ready!')
"
        
        # Create data directory
        New-Item -ItemType Directory -Path "data" -Force | Out-Null
        
        # Create apps/__init__.py if missing
        if (-not (Test-Path "apps/__init__.py")) {
            Write-Host "📝 Creating apps package init..." -ForegroundColor Yellow
            "" | Out-File -FilePath "apps/__init__.py" -Encoding UTF8
        }
        if (-not (Test-Path "apps/backend/__init__.py")) {
            New-Item -ItemType Directory -Path "apps/backend" -Force | Out-Null
            "" | Out-File -FilePath "apps/backend/__init__.py" -Encoding UTF8
        }
        
        # Run tests
        Write-Host "🧪 Running tests..." -ForegroundColor Yellow
        python -m pytest tests/ -q
        
        # Set environment variables
        $env:API_KEY = $ApiKey
        
        # Start backend in background
        Write-Host "🖥️  Starting FastAPI backend..." -ForegroundColor Yellow
        $BackendJob = Start-Job -ScriptBlock {
            param($ApiKey)
            $env:API_KEY = $ApiKey
            Set-Location $using:PWD
            .\.venv\Scripts\Activate.ps1
            uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000
        } -ArgumentList $ApiKey
        
        # Wait for backend to start
        Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
        
        # Test backend health
        $MaxRetries = 5
        $Retry = 0
        $BackendReady = $false
        
        do {
            try {
                $Retry++
                Write-Host "🔍 Testing backend connection (attempt $Retry)..." -ForegroundColor Gray
                $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
                if ($response.status -eq "healthy") {
                    $BackendReady = $true
                    Write-Host "✅ Backend health check passed" -ForegroundColor Green
                }
            } catch {
                Write-Host "⏳ Backend not ready yet..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
            }
        } while (-not $BackendReady -and $Retry -lt $MaxRetries)
        
        if (-not $BackendReady) {
            Write-Host "❌ Backend failed to start after $MaxRetries attempts" -ForegroundColor Red
            Stop-Job $BackendJob
            Remove-Job $BackendJob
            exit 1
        }
        
        # Test authenticated endpoint
        try {
            $headers = @{"X-API-KEY" = $ApiKey}
            $statusResponse = Invoke-RestMethod -Uri "http://localhost:8000/status" -Headers $headers
            Write-Host "✅ Backend authentication test passed" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ Backend authentication test failed" -ForegroundColor Yellow
        }
        
        # Start frontend in API mode
        $env:BACKEND_URL = "http://localhost:8000"
        $env:API_KEY = $ApiKey
        
        Write-Host ""
        Write-Host "🎯 Starting Streamlit (API Mode)..." -ForegroundColor Green
        Write-Host "🖥️  Backend: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "🌐 Frontend: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "🔑 API Key: $($ApiKey.Substring(0,8))..." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Press Ctrl+C to stop both services" -ForegroundColor Gray
        
        try {
            streamlit run apps/frontend/demo_streamlit.py
        } finally {
            Write-Host "🛑 Stopping backend..." -ForegroundColor Yellow
            Stop-Job $BackendJob -ErrorAction SilentlyContinue
            Remove-Job $BackendJob -ErrorAction SilentlyContinue
        }
    }
    
    "docker" {
        Write-Host "🐳 Starting with Docker Compose..." -ForegroundColor Green
        Write-Host "This mode builds and runs everything in containers" -ForegroundColor Gray
        Write-Host ""
        
        # Check if Docker is installed
        try {
            docker --version | Out-Null
            Write-Host "✅ Docker is installed" -ForegroundColor Green
        } catch {
            Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
            Write-Host "Please install Docker Desktop from https://docker.com" -ForegroundColor Yellow
            exit 1
        }
        
        # Check if Docker is running
        try {
            docker info | Out-Null
            Write-Host "✅ Docker is running" -ForegroundColor Green
        } catch {
            Write-Host "❌ Docker is not running" -ForegroundColor Red
            Write-Host "Please start Docker Desktop" -ForegroundColor Yellow
            exit 1
        }
        
        # Set secure API key for containers
        $env:API_KEY = $ApiKey
        
        Write-Host "🔨 Building and starting containers..." -ForegroundColor Yellow
        Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray
        
        # Build and start with docker-compose
        docker-compose up --build
        
        Write-Host ""
        Write-Host "🌐 Services should be available at:" -ForegroundColor Cyan
        Write-Host "   Backend: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "   Frontend: http://localhost:8501" -ForegroundColor Cyan
    }
    
    "test" {
        Write-Host "🧪 Running Test Suite..." -ForegroundColor Green
        Write-Host ""
        
        # Setup environment
        if (-not (Test-Path ".venv")) {
            python -m venv .venv
        }
        .\.venv\Scripts\Activate.ps1
        pip install --upgrade pip -q
        pip install -r requirements.txt -q
        
        # Create missing package files
        if (-not (Test-Path "apps/__init__.py")) {
            "" | Out-File -FilePath "apps/__init__.py" -Encoding UTF8
        }
        if (-not (Test-Path "apps/backend/__init__.py")) {
            New-Item -ItemType Directory -Path "apps/backend" -Force | Out-Null
            "" | Out-File -FilePath "apps/backend/__init__.py" -Encoding UTF8
        }
        
        # Pre-warm model for tests
        Write-Host "🤖 Pre-warming model for tests..." -ForegroundColor Yellow
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" 2>$null
        
        # Run comprehensive tests
        Write-Host "🧪 Running comprehensive test suite..." -ForegroundColor Yellow
        python -m pytest tests/ -v --tb=short
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ All tests passed! System is ready for deployment." -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "❌ Some tests failed. Check output above." -ForegroundColor Red
            exit 1
        }
    }
    
    default {
        Write-Host "❌ Unknown mode: $Mode" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available modes:" -ForegroundColor Yellow
        Write-Host "  local  - Run Streamlit with embedded processor" -ForegroundColor Gray
        Write-Host "  api    - Run FastAPI backend + Streamlit frontend" -ForegroundColor Gray  
        Write-Host "  docker - Run everything in Docker containers" -ForegroundColor Gray
        Write-Host "  test   - Run comprehensive test suite" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Example: .\run_demo.ps1 -Mode api -ApiKey your-secret-key" -ForegroundColor Gray
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Demo completed!" -ForegroundColor Green
