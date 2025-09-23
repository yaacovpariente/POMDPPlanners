#!/bin/bash
# mock_docker_tests.sh - Mock Docker-based CI/CD tests locally

echo "🐳 Mocking Docker-based CI/CD tests locally..."

# Mock: Run linting inside Docker
echo "📝 Running pylint (mocking Docker step)..."
pylint POMDPPlanners/ --exit-zero

# Mock: Run code formatting check inside Docker  
echo "🎨 Running black check (mocking Docker step)..."
black --check POMDPPlanners/

# Mock: Run type checking inside Docker
echo "🔍 Running mypy (mocking Docker step)..."
mypy POMDPPlanners/ --ignore-missing-imports

# Mock: Run unit tests inside Docker
echo "🧪 Running pytest with coverage (mocking Docker step)..."
pytest POMDPPlanners/tests/ -v --cov=POMDPPlanners --cov-report=xml --cov-report=term-missing

# Mock: Run doctests inside Docker
echo "📚 Running doctests (mocking Docker step)..."
pytest --doctest-modules POMDPPlanners/ -v

# Mock: Run visualization examples inside Docker
echo "📊 Running visualization examples (mocking Docker step)..."
export MPLBACKEND=Agg
python legacy/light_dark_pomdp_visualization_example.py
python legacy/pomcpow_rock_sample_visualization_demo.py
python legacy/push_pomdp_visualization_example.py

# Mock: Build documentation inside Docker
echo "📖 Building documentation (mocking Docker step)..."
cd docs
sphinx-build -b html . _build/html
cd ..

echo "✅ All Docker-based CI/CD tests completed locally!"
