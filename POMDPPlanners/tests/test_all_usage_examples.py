#!/usr/bin/env python3
"""Master test script to run all usage examples from environments, planners, and simulations."""

import sys
import os
import traceback
from pathlib import Path
import random
import numpy as np

np.random.seed(42)
random.seed(42)


# Add the current directory to Python path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_test_module(module_path: str, test_name: str):
    """Run a test module and return the result."""
    print(f"=" * 60)
    print(f"Running {test_name}")
    print(f"=" * 60)
    
    try:
        # Import and run the test module
        if module_path == "test_environments.test_all_environment_examples":
            from test_environments.test_all_environment_examples import main
        elif module_path == "test_planners.test_planners_usage_examples":
            from test_planners.test_planners_usage_examples import main
        elif module_path == "test_simulations.test_simulations_usage_examples":
            from test_simulations.test_simulations_usage_examples import main
        else:
            raise ImportError(f"Unknown module path: {module_path}")
        
        result = main()
        print(f"\n{test_name} completed with result: {result}")
        return result == 0
        
    except Exception as e:
        print(f"\n❌ {test_name} failed to run: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all usage example tests across the entire codebase."""
    print("🚀 Running ALL usage examples from the entire POMDPPlanners codebase")
    print("This comprehensive test validates all documented examples work correctly.\n")
    
    # Define all test modules to run
    test_modules = [
        ("test_environments.test_all_environment_examples", "Environment Usage Examples"),
        ("test_planners.test_planners_usage_examples", "Planner Usage Examples"), 
        ("test_simulations.test_simulations_usage_examples", "Simulation Usage Examples"),
    ]
    
    results = []
    passed_modules = 0
    total_modules = len(test_modules)
    
    # Run each test module
    for module_path, test_name in test_modules:
        success = run_test_module(module_path, test_name)
        results.append((test_name, success))
        if success:
            passed_modules += 1
        print()  # Add spacing between modules
    
    # Print comprehensive summary
    print("=" * 80)
    print("COMPREHENSIVE USAGE EXAMPLES TEST RESULTS")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:<12} {test_name}")
    
    print("-" * 80)
    print(f"OVERALL RESULT: {passed_modules}/{total_modules} test modules passed")
    
    if passed_modules == total_modules:
        print("\n🎉 SUCCESS: ALL usage examples across the entire codebase work correctly!")
        print("\nThis means:")
        print("• All environment classes have working usage examples")
        print("• All planner classes have working usage examples") 
        print("• All simulation classes have working usage examples")
        print("• The documentation is accurate and examples are runnable")
        print("• New users can copy-paste examples and they will work")
        return 0
    else:
        failed_count = total_modules - passed_modules
        print(f"\n❌ FAILURE: {failed_count} test module(s) have issues that need attention.")
        print("\nFailed modules may have:")
        print("• Outdated usage examples in docstrings")
        print("• Import errors or missing dependencies")
        print("• Bugs in the example code")
        print("• Changes in API that broke existing examples")
        return 1

if __name__ == "__main__":
    exit(main())