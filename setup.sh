#!/bin/bash

echo "🔧 Setting up Docker RAG Test Environment..."
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo ""
else
    echo "✅ Homebrew already installed"
fi

# Install Docker components
echo "🐳 Installing Docker components..."

# Check what Docker components are needed
NEED_DOCKER=false
NEED_COMPOSE=false
NEED_COLIMA=false

if ! command -v docker &> /dev/null; then
    NEED_DOCKER=true
fi

if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    NEED_COMPOSE=true
fi

if ! command -v colima &> /dev/null; then
    NEED_COLIMA=true
fi

# Install what's needed
if [ "$NEED_COLIMA" = true ] || [ "$NEED_DOCKER" = true ]; then
    echo "Installing Colima and Docker CLI..."
    brew install colima docker docker-buildx
    echo ""
fi

if [ "$NEED_COMPOSE" = true ]; then
    echo "Installing Docker Compose..."
    brew install docker-compose
    echo ""
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv for faster installation..."
    uv pip install -r requirements.txt
else
    echo "Installing uv for faster Python package management..."
    brew install uv
    uv pip install -r requirements.txt
fi
echo ""

# Install credential helper
echo "🔐 Installing Docker credential helper..."
brew install docker-credential-helper
echo ""

# Fix Docker config if it exists
if [ -f ~/.docker/config.json ]; then
    echo "🔧 Fixing Docker credential configuration..."
    # Back up the config
    cp ~/.docker/config.json ~/.docker/config.json.bak
    
    # Remove problematic credsStore if it exists
    if command -v jq &> /dev/null; then
        jq 'del(.credsStore) | .credsStore = "osxkeychain"' ~/.docker/config.json > /tmp/config.json && mv /tmp/config.json ~/.docker/config.json
    else
        echo "Installing jq for JSON processing..."
        brew install jq
        jq 'del(.credsStore) | .credsStore = "osxkeychain"' ~/.docker/config.json > /tmp/config.json && mv /tmp/config.json ~/.docker/config.json
    fi
    echo ""
fi

# Start Colima if it was installed
if command -v colima &> /dev/null; then
    echo "🚀 Starting Colima Docker runtime..."
    colima stop 2>/dev/null || true
    colima start
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "Now you can run:"
echo "  ./start_frontend.sh"
echo ""
echo "This will start:"
echo "  - RAG API: http://localhost:8000"
echo "  - Streamlit UI: http://localhost:8501"