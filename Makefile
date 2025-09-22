# POMDPPlanners Makefile
# Provides convenient targets for development and testing

.PHONY: help test-docs test-docs-quick test-docs-verbose build-docs clean-docs install-dev

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test-docs: ## Run documentation tests
	python test_documentation.py

test-docs-quick: ## Run quick documentation tests
	python test_documentation.py --quick

test-docs-verbose: ## Run documentation tests with verbose output
	python test_documentation.py --verbose

build-docs: ## Build documentation
	./build_docs.sh

clean-docs: ## Clean documentation build artifacts
	rm -rf docs/_build/
	rm -rf docs/api/

test-all: test-docs ## Run all tests (placeholder for future expansion)
	@echo "All tests completed"

ci-test: ## Run tests as they would run in CI
	python test_documentation.py --verbose
	./build_docs.sh

pre-commit: test-docs-quick ## Run pre-commit checks
	@echo "Pre-commit checks completed"
