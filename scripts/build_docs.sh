#!/bin/bash

# Build documentation script for POMDPPlanners
# This script builds the Sphinx documentation locally

set -e

echo "🔧 Building POMDPPlanners Documentation"
echo "======================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   It's recommended to run this in a virtual environment."
    echo "   Run: source .venv/bin/activate"
    echo ""
fi

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Error: pyproject.toml not found."
    echo "   Please run this script from the project root directory."
    exit 1
fi

# Install the package with documentation dependencies
echo "📦 Installing POMDPPlanners with documentation dependencies..."
pip install -e ".[docs]"

# Navigate to docs directory
cd docs

echo "🏗️  Generating API documentation..."
sphinx-apidoc -o api ../POMDPPlanners tests --force --module-first

echo "🏗️  Building HTML documentation..."
sphinx-build -b html . _build/html

echo "✅ Documentation built successfully!"
echo ""
echo "📖 Open docs/_build/html/index.html in your browser to view the documentation"
echo "   Or run: python -m http.server 8000 -d _build/html"
echo ""
echo "ℹ️  Note: GitHub Pages deployment requires a public repository"
echo "   Documentation will auto-deploy when the repository becomes public"
echo ""
echo "🚀 For automatic rebuilding during development, install sphinx-autobuild:"
echo "   pip install sphinx-autobuild"
echo "   sphinx-autobuild . _build/html --open-browser"