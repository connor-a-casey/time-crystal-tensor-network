#!/bin/bash


echo "Test Runner Script"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import tenpy, numpy, matplotlib, scipy, tqdm, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing from requirements.txt..."
    pip install -r requirements.txt
fi

echo ""
echo "Running all tests..."
echo ""

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0"
    echo ""
    echo "Runs the complete test suite including:"
    echo "  - Basic functionality tests (21 tests)"
    echo "  - Physics validation tests (16 tests)"  
    echo "  - Performance benchmarks (12 tests)"
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0                           # Run all tests"
else
    # Run all tests
    echo "Running all tests (basic functionality + physics validation + performance)..."
    python tests/run_tests.py
fi

echo ""
echo "Test run complete!"