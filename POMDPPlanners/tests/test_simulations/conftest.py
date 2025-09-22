import pytest
import mlflow
import os
import random
import numpy as np

np.random.seed(42)
random.seed(42)


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """Automatically cleanup any active MLflow runs before and after each test.

    This fixture ensures that MLflow runs are properly ended between tests,
    preventing the "Run already active" error that occurs when tests don't
    properly clean up their MLflow runs.
    """
    # Before test: end any existing runs
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        # Ignore any errors during cleanup
        pass

    yield

    # After test: end any remaining runs
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        # Ignore any errors during cleanup
        pass
