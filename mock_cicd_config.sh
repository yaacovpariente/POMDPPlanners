# mock_cicd_config.sh - Configuration for mocking CI/CD tests

# Enable/disable specific test categories
ENABLE_DOCKER_TESTS=true
ENABLE_MULTI_PYTHON_TESTS=true
ENABLE_PACKAGE_TESTS=true
ENABLE_DOCS_TESTS=true
ENABLE_COVERAGE_TESTS=true

# Python versions to test (if available)
PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12")

# Test parameters
PYTEST_ARGS="-v"
COVERAGE_ARGS="--cov=POMDPPlanners --cov-report=xml --cov-report=term-missing"
PYLINT_ARGS="--exit-zero"
BLACK_ARGS="--check"
MYPY_ARGS="--ignore-missing-imports"

# Optional tests (continue on error)
ENABLE_VISUALIZATION_TESTS=true
ENABLE_DOCS_BUILD=true

# Mock environment variables
export MPLBACKEND=Agg
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CI=true  # Simulate CI environment
export GITHUB_ACTIONS=true  # Simulate GitHub Actions environment
