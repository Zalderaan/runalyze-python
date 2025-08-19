#!/bin/bash
# Render startup script for RunAnalyze API

# Create necessary directories
mkdir -p tmp
mkdir -p debug

# Set environment for production
export ENVIRONMENT=production

# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 300
