#!/bin/bash

# Start the RAG app with Streamlit frontend
echo "🚀 Starting RAG App with Streamlit Frontend..."
echo ""

# Check if colima is installed and start it if needed
if command -v colima &> /dev/null; then
    echo "📦 Starting Colima Docker runtime..."
    colima start 2>/dev/null || echo "ℹ️  Colima already running"
    echo ""
fi

# Check Docker connection
echo "🔍 Checking Docker connection..."
if ! docker info &> /dev/null; then
    echo "❌ Cannot connect to Docker daemon!"
    echo "Try:"
    echo "  colima restart"
    echo "  docker context use colima"
    exit 1
fi
echo "✅ Docker connection OK"
echo ""

echo "Services starting:"
echo "  - RAG API: http://localhost:8000"
echo "  - Streamlit UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Disable Docker BuildKit to avoid buildx issues
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Set Docker context to colima if available
if command -v colima &> /dev/null && colima status &> /dev/null; then
    docker context use colima 2>/dev/null || true
fi

# Build and start services (try modern syntax first, fallback to legacy)
if docker compose version &> /dev/null; then
    echo "Using docker compose..."
    docker compose up --build
elif command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose up --build
else
    echo "❌ Docker Compose not found!"
    echo ""
    echo "Please install Docker Compose:"
    echo "  brew install docker-compose"
    echo ""
    echo "Or if you have newer Docker, try:"
    echo "  brew upgrade docker"
    exit 1
fi