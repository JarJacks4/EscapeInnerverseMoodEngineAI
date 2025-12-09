#!/bin/bash

# Escape API Test Runner
# This script starts the API server and runs comprehensive tests

echo "🚀 ESCAPE API TEST RUNNER"
echo "========================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Install requests if needed
echo "📦 Installing test dependencies..."
pip install requests > /dev/null 2>&1

echo "🔧 Starting API server..."
# Start the API server in background
python main.py &
API_PID=$!

echo "🔍 API server started with PID: $API_PID"
echo "⏳ Waiting for server to initialize..."
sleep 5

echo ""
echo "🧪 Running comprehensive tests..."
echo "================================"

# Run the test suite
python test_api.py

# Clean up
echo ""
echo "🧹 Cleaning up..."
kill $API_PID 2>/dev/null
echo "✅ Test run complete!"