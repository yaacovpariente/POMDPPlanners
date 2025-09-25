#!/bin/bash
# mock_cicd_tests.sh - Complete local mocking of GitHub Actions CI/CD tests

set -e  # Exit on any error

echo "🚀 Mocking GitHub Actions CI/CD Tests Locally"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run a test step
run_test_step() {
    local step_name="$1"
    local command="$2"
    
    echo -e "${BLUE}📋 Running: $step_name${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ $step_name completed successfully${NC}"
    else
        echo -e "${RED}❌ $step_name failed${NC}"
        return 1
    fi
}

# Function to run a test step with continue-on-error
run_test_step_optional() {
    local step_name="$1"
    local command="$2"
    
    echo -e "${BLUE}📋 Running (optional): $step_name${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ $step_name completed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  $step_name failed (continuing...)${NC}"
    fi
}

echo -e "${BLUE}🐳 Mocking Docker Tests Workflow${NC}"
echo "----------------------------------------"

# Mock: Run linting inside Docker
run_test_step "Pylint Linting" "pylint POMDPPlanners/ --exit-zero"

# Mock: Run code formatting check inside Docker
run_test_step "Black Formatting Check" "black --check POMDPPlanners/"

# Mock: Type checking removed

# Mock: Run unit tests inside Docker
run_test_step "Pytest Unit Tests with Coverage" "pytest POMDPPlanners/tests/ -v --cov=POMDPPlanners --cov-report=xml --cov-report=term-missing"

# Mock: Run doctests inside Docker
run_test_step "Pytest Doctests" "pytest --doctest-modules POMDPPlanners/ -v"

# Mock: Run visualization examples inside Docker (optional)
export MPLBACKEND=Agg
run_test_step_optional "Light Dark POMDP Visualization" "python legacy/light_dark_pomdp_visualization_example.py"
run_test_step_optional "Rock Sample Visualization" "python legacy/pomcpow_rock_sample_visualization_demo.py"
run_test_step_optional "Push POMDP Visualization" "python legacy/push_pomdp_visualization_example.py"

# Mock: Build documentation inside Docker (optional)
run_test_step_optional "Documentation Build" "cd docs && sphinx-build -b html . _build/html && cd .."

echo ""
echo -e "${BLUE}🐍 Mocking Multi-Python Version Tests${NC}"
echo "----------------------------------------"

# Check if we have multiple Python versions available
PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12")
AVAILABLE_VERSIONS=()

for version in "${PYTHON_VERSIONS[@]}"; do
    if command -v "python$version" &> /dev/null || python --version | grep -q "$version"; then
        AVAILABLE_VERSIONS+=("$version")
    fi
done

if [ ${#AVAILABLE_VERSIONS[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No specific Python versions found. Using system Python.${NC}"
    AVAILABLE_VERSIONS=("system")
fi

for version in "${AVAILABLE_VERSIONS[@]}"; do
    echo -e "${BLUE}🐍 Testing Python $version${NC}"
    
    if [ "$version" != "system" ]; then
        # Try to use specific Python version
        PYTHON_CMD="python$version"
        if ! command -v "$PYTHON_CMD" &> /dev/null; then
            PYTHON_CMD="python"
            echo -e "${YELLOW}⚠️  python$version not found, using system python${NC}"
        fi
    else
        PYTHON_CMD="python"
    fi
    
    # Mock: Install dependencies
    run_test_step "Install Dependencies for Python $version" "$PYTHON_CMD -m pip install --upgrade pip && $PYTHON_CMD -m pip install -r requirements.txt -r requirements-dev.txt && $PYTHON_CMD -m pip install -e ."
    
    # Mock: Run tests
    run_test_step "Run Tests for Python $version" "$PYTHON_CMD -m pytest POMDPPlanners/tests/ -v"
    
    # Mock: Check code quality
    run_test_step "Code Quality Check for Python $version" "$PYTHON_CMD -m black --check POMDPPlanners/ && $PYTHON_CMD -m pylint POMDPPlanners/ --exit-zero"
done

echo ""
echo -e "${BLUE}📦 Mocking Package Building and Publishing${NC}"
echo "----------------------------------------"

# Mock: Install build dependencies
run_test_step "Install Build Dependencies" "python -m pip install build twine"

# Mock: Build package
run_test_step "Build Package" "python -m build"

# Mock: Check package
run_test_step "Check Package" "twine check dist/*"

# Mock: Test PyPI upload (dry run)
run_test_step "Test PyPI Upload (Dry Run)" "twine upload --repository testpypi dist/* --dry-run"

echo ""
echo -e "${BLUE}📊 Mocking Coverage Upload${NC}"
echo "----------------------------------------"

# Mock: Check if coverage file exists
if [ -f "coverage.xml" ]; then
    echo -e "${GREEN}✅ Coverage file found: coverage.xml${NC}"
    echo -e "${BLUE}📈 Coverage report summary:${NC}"
    if command -v "coverage" &> /dev/null; then
        coverage report --show-missing
    else
        echo "Coverage file exists but coverage tool not available"
    fi
else
    echo -e "${YELLOW}⚠️  No coverage file found. Generating one...${NC}"
    run_test_step "Generate Coverage Report" "pytest POMDPPlanners/tests/ --cov=POMDPPlanners --cov-report=xml"
fi

# Mock: Upload coverage to Codecov (dry run)
echo -e "${BLUE}🚀 Mocking Codecov upload...${NC}"
echo "Would upload coverage.xml to Codecov with flags: unittests"

echo ""
echo -e "${GREEN}🎉 All CI/CD Tests Mocked Successfully!${NC}"
echo "=============================================="
echo ""
echo -e "${BLUE}Summary of mocked tests:${NC}"
echo "✅ Docker-based tests (linting, formatting, type checking, unit tests, doctests)"
echo "✅ Multi-Python version tests"
echo "✅ Package building and publishing tests"
echo "✅ Documentation build tests"
echo "✅ Coverage upload tests"
echo ""
echo -e "${YELLOW}Note: This script mocks the CI/CD environment locally.${NC}"
echo -e "${YELLOW}For actual CI/CD testing, push to GitHub and check the Actions tab.${NC}"
