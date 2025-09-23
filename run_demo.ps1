# AI Plagiarism Detection System - Production Demo Script
Write-Host "🔍 Setting up AI Plagiarism Detection System..." -ForegroundColor Green

# Check if venv exists, create if not
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "⚡ Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "🔧 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements with compatibility fixes
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
pip install huggingface_hub==0.25.2
pip install sentence-transformers==2.2.2
pip install -r requirements.txt

# Pre-warm AI model (prevents demo delays)
Write-Host "🤖 Pre-warming AI model (prevents demo stalls)..." -ForegroundColor Yellow
python -c "
from sentence_transformers import SentenceTransformer
print('🔥 Caching AI model for instant demo...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✅ AI model ready for flawless demonstration!')
"

# Create data directory
Write-Host "📁 Creating data directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "data" -Force | Out-Null

# Run tests to verify system
Write-Host "🧪 Running system tests..." -ForegroundColor Yellow
python -m pytest tests/ -v

# Start the AI application
Write-Host "🚀 Launching AI Plagiarism Detection System..." -ForegroundColor Green
Write-Host "🌐 Demo available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "✅ AI model pre-warmed - Zero loading delays during demo!" -ForegroundColor Green
streamlit run apps/frontend/demo_streamlit.py
