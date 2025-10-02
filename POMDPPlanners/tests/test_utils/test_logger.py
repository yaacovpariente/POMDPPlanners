import logging
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from POMDPPlanners.utils.logger import (
    cleanup_all_loggers,
    get_logger,
    get_queue_logger_diagnostics,
    get_queue_logger_manager,
)

np.random.seed(42)
random.seed(42)


def test_logger_console_output_true(capsys):
    """Test logger console output true.

    Purpose: Validates that logger correctly outputs messages to console when console_output=True

    Given: Logger configured with debug=True and console_output=True
    When: Info message is logged
    Then: Message appears in captured stdout or stderr output

    Test type: unit
    """
    logger = get_logger("test.console", debug=True, console_output=True)
    logger.info("This is a test message for console output.")
    captured = capsys.readouterr()
    assert "This is a test message for console output." in captured.out or captured.err


def test_logger_console_output_false(capsys):
    """Test logger console output false.

    Purpose: Validates that logger does not output messages to console when console_output=False

    Given: Logger configured with debug=True and console_output=False
    When: Info message is logged
    Then: Message does not appear in captured stdout or stderr output

    Test type: unit
    """
    logger = get_logger("test.no_console", debug=True, console_output=False)
    logger.info("This should not appear in the console.")
    captured = capsys.readouterr()
    assert "This should not appear in the console." not in captured.out
    assert "This should not appear in the console." not in captured.err


def test_logger_file_output(tmp_path):
    """Test logger file output.

    Purpose: Validates that logger correctly writes messages to log files when output_dir is specified

    Given: Logger configured with debug=True, output_dir=tmp_path, and console_output=False
    When: Info message is logged
    Then: Log file is created in logs subdirectory containing the test message

    Test type: unit
    """
    log_dir = tmp_path / "logs"
    logger = get_logger("test.file", debug=True, output_dir=tmp_path, console_output=False)
    logger.info("This is a test message for file output.")
    # Find the log file
    log_files = list(log_dir.glob("test_file_*.log"))
    assert len(log_files) == 1
    with open(log_files[0], "r") as f:
        content = f.read()
    assert "This is a test message for file output." in content


def test_logger_no_duplicate_handlers(tmp_path):
    """Test logger no duplicate handlers.

    Purpose: Validates that repeated get_logger calls do not create duplicate handlers causing message duplication

    Given: Logger called twice with identical parameters (debug=True, output_dir=tmp_path, console_output=True)
    When: Messages are logged after each get_logger call
    Then: Each message appears exactly once in the log file without duplication

    Test type: unit
    """
    logger = get_logger("test.duplicate", debug=True, output_dir=tmp_path, console_output=True)
    logger.info("First message.")
    # Call get_logger again to simulate repeated calls
    logger = get_logger("test.duplicate", debug=True, output_dir=tmp_path, console_output=True)
    logger.info("Second message.")
    # Should not duplicate messages in the log file
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("test_duplicate_*.log"))
    assert len(log_files) == 1
    with open(log_files[0], "r") as f:
        content = f.read()
    # Each message should appear only once
    assert content.count("First message.") == 1
    assert content.count("Second message.") == 1


def test_queue_logger_basic_functionality(tmp_path):
    """Test basic queue-based logger functionality.

    Purpose: Validates that queue-based logger works correctly for basic logging operations

    Given: Logger configured with use_queue=True, output_dir=tmp_path, console_output=False
    When: Info and warning messages are logged
    Then: Messages are written to log file via queue mechanism without blocking

    Test type: unit
    """
    try:
        logger = get_logger(
            name="test.queue.basic",
            debug=False,
            output_dir=tmp_path,
            console_output=False,
            use_queue=True,
        )

        # Log some messages
        logger.info("Queue logger test message 1")
        logger.warning("Queue logger test warning")
        logger.info("Queue logger test message 2")

        # Give writer thread time to process
        time.sleep(1.0)

        # Check that log file was created
        log_dir = tmp_path / "logs"
        assert log_dir.exists()

        log_files = list(log_dir.glob("test_queue_basic_*.log"))
        assert len(log_files) == 1

        # Check content
        with open(log_files[0], "r") as f:
            content = f.read()

        assert "Queue logger test message 1" in content
        assert "Queue logger test warning" in content
        assert "Queue logger test message 2" in content

    finally:
        cleanup_all_loggers()


def test_queue_logger_individual_task_files(tmp_path):
    """Test queue logger creates individual files per task.

    Purpose: Validates that queue-based logger maintains individual log files for different tasks

    Given: Multiple queue loggers with different task IDs
    When: Messages are logged from each task
    Then: Separate log files are created for each task with correct content

    Test type: unit
    """
    try:
        # Create multiple task loggers
        loggers = []
        task_names = []

        for i in range(3):
            task_name = f"test.task.{i}.task_id.abc{i:03d}"
            logger = get_logger(
                name=task_name,
                debug=False,
                output_dir=tmp_path,
                console_output=False,
                use_queue=True,
            )
            loggers.append(logger)
            task_names.append(task_name)

        # Log from each task
        for i, logger in enumerate(loggers):
            logger.info(f"Message from task {i}")
            logger.warning(f"Warning from task {i}")

        # Give writer thread time to process
        time.sleep(1.0)

        # Check that separate log files were created
        log_dir = tmp_path / "logs"
        assert log_dir.exists()

        log_files = list(log_dir.glob("test_task_*_task_id_*.log"))
        assert len(log_files) == 3

        # Check each file has correct content
        for i in range(3):
            matching_files = [f for f in log_files if f"task_{i}_task_id_abc{i:03d}" in f.name]
            assert len(matching_files) == 1

            with open(matching_files[0], "r") as f:
                content = f.read()

            assert f"Message from task {i}" in content
            assert f"Warning from task {i}" in content

    finally:
        cleanup_all_loggers()


def test_queue_logger_diagnostics(tmp_path):
    """Test queue logger diagnostics functionality.

    Purpose: Validates that diagnostic functions provide accurate information about queue logger state

    Given: Queue logger system with multiple active loggers
    When: Diagnostics are requested
    Then: Accurate information about queue size, handlers, and system state is returned

    Test type: unit
    """
    try:
        # Create some queue loggers
        logger1 = get_logger(
            name="test.diag.1",
            output_dir=tmp_path,
            console_output=False,
            use_queue=True,
        )
        logger2 = get_logger(
            name="test.diag.2",
            output_dir=tmp_path,
            console_output=False,
            use_queue=True,
        )

        # Log some messages
        logger1.info("Test message 1")
        logger2.info("Test message 2")

        # Give writer thread time to process
        time.sleep(1.0)

        # Get diagnostics
        diagnostics = get_queue_logger_diagnostics()

        # Check diagnostic information
        assert isinstance(diagnostics, dict)
        assert "queue_size" in diagnostics
        assert "writer_thread_alive" in diagnostics
        assert "registered_loggers" in diagnostics
        assert "active_handlers" in diagnostics
        assert "max_handlers" in diagnostics
        assert "shutdown_event_set" in diagnostics

        # Verify some values
        assert diagnostics["writer_thread_alive"] is True
        assert diagnostics["registered_loggers"] >= 2
        assert diagnostics["shutdown_event_set"] is False

    finally:
        cleanup_all_loggers()
        time.sleep(0.5)  # Give cleanup time to complete

        # Check diagnostics after cleanup
        final_diagnostics = get_queue_logger_diagnostics()
        assert final_diagnostics["registered_loggers"] == 0
        assert final_diagnostics["active_handlers"] == 0


def test_queue_logger_cleanup():
    """Test queue logger cleanup functionality.

    Purpose: Validates that cleanup functions properly shut down queue logger resources

    Given: Active queue logger with writer thread and handlers
    When: cleanup_all_loggers is called
    Then: All resources are properly cleaned up and system returns to clean state

    Test type: unit
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create queue logger
        logger = get_logger(name="test.cleanup", output_dir=tmp_path, use_queue=True)

        logger.info("Test cleanup message")

        # Verify system is active
        diagnostics_before = get_queue_logger_diagnostics()
        assert diagnostics_before["writer_thread_alive"] is True
        assert diagnostics_before["registered_loggers"] > 0

        # Cleanup
        cleanup_all_loggers()

        # Verify cleanup worked
        diagnostics_after = get_queue_logger_diagnostics()
        assert diagnostics_after["registered_loggers"] == 0
        assert diagnostics_after["active_handlers"] == 0


def test_queue_logger_handler_management(tmp_path):
    """Test queue logger handler pooling and management.

    Purpose: Validates that queue logger properly manages file handlers with pooling

    Given: Multiple queue loggers accessing the same task repeatedly
    When: Handlers are created and reused
    Then: Handler count is managed efficiently without resource leaks

    Test type: unit
    """
    try:
        task_name = "test.handler.pooling.task_id.xyz123"

        # Create multiple loggers for same task
        logger1 = get_logger(name=task_name, output_dir=tmp_path, use_queue=True)
        logger2 = get_logger(name=task_name, output_dir=tmp_path, use_queue=True)

        # They should be the same logger instance
        assert logger1.name == logger2.name

        # Log messages
        logger1.info("Message from logger1")
        logger2.info("Message from logger2")

        time.sleep(1.0)

        # Check diagnostics
        diagnostics = get_queue_logger_diagnostics()

        # Should have reasonable handler count
        assert diagnostics["active_handlers"] <= diagnostics["max_handlers"]
        assert diagnostics["active_handlers"] >= 1

        # Check log file
        log_dir = tmp_path / "logs"
        log_files = list(log_dir.glob("test_handler_pooling_task_id_*.log"))
        assert len(log_files) == 1

        with open(log_files[0], "r") as f:
            content = f.read()

        assert "Message from logger1" in content
        assert "Message from logger2" in content

    finally:
        cleanup_all_loggers()


def test_queue_logger_backwards_compatibility(tmp_path):
    """Test queue logger backwards compatibility with existing code.

    Purpose: Validates that queue logger maintains compatibility with existing get_logger calls

    Given: Logger calls with and without use_queue parameter
    When: Both types of loggers are used
    Then: Both work correctly and produce expected log files

    Test type: integration
    """
    try:
        # Test individual logger (default behavior)
        logger_individual = get_logger(
            name="test.compat.individual", output_dir=tmp_path, console_output=False
        )

        # Test queue logger (new behavior)
        logger_queue = get_logger(
            name="test.compat.queue",
            output_dir=tmp_path,
            console_output=False,
            use_queue=True,
        )

        # Log messages
        logger_individual.info("Individual logger message")
        logger_queue.info("Queue logger message")

        time.sleep(1.0)

        # Check log files
        log_dir = tmp_path / "logs"
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 2

        # Find and check individual logger file
        individual_files = [f for f in log_files if "test_compat_individual" in f.name]
        assert len(individual_files) == 1

        with open(individual_files[0], "r") as f:
            content = f.read()
        assert "Individual logger message" in content

        # Find and check queue logger file
        queue_files = [f for f in log_files if "test_compat_queue" in f.name]
        assert len(queue_files) == 1

        with open(queue_files[0], "r") as f:
            content = f.read()
        assert "Queue logger message" in content

    finally:
        cleanup_all_loggers()


def test_logger_no_multiple_files_on_repeated_calls(tmp_path):
    """Test that repeated get_logger calls with same name don't create multiple log files.

    Purpose: Validates that repeated calls to get_logger with same name reuse existing logger
             and don't create new log files with different timestamps

    Given: Multiple sequential calls to get_logger with the same logger name
    When: Messages are logged after each get_logger call
    Then: Only ONE log file is created, containing all messages from all calls

    Test type: unit
    """
    logger_name = "test.single_file"

    # First call - creates logger and log file
    logger1 = get_logger(logger_name, debug=True, output_dir=tmp_path, console_output=False)
    logger1.info("Message from first call")

    # Small delay to ensure timestamp would be different if new file was created
    time.sleep(0.1)

    # Second call - should reuse existing logger and file
    logger2 = get_logger(logger_name, debug=True, output_dir=tmp_path, console_output=False)
    logger2.info("Message from second call")

    # Small delay
    time.sleep(0.1)

    # Third call - should still reuse existing logger and file
    logger3 = get_logger(logger_name, debug=True, output_dir=tmp_path, console_output=False)
    logger3.info("Message from third call")

    # Check that only ONE log file was created
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("test_single_file_*.log"))

    assert len(log_files) == 1, (
        f"Expected exactly 1 log file, but found {len(log_files)}: "
        f"{[f.name for f in log_files]}"
    )

    # Check that all messages are in the single log file
    with open(log_files[0], "r") as f:
        content = f.read()

    assert "Message from first call" in content
    assert "Message from second call" in content
    assert "Message from third call" in content

    # Verify messages appear exactly once
    assert content.count("Message from first call") == 1
    assert content.count("Message from second call") == 1
    assert content.count("Message from third call") == 1


def test_logger_reuses_handlers_with_same_name(tmp_path):
    """Test that get_logger with same name returns logger with existing handlers.

    Purpose: Validates that calling get_logger multiple times with the same name
             returns the same logger instance with the same handlers

    Given: Logger created with specific name and output directory
    When: get_logger is called again with the same name
    Then: Same logger instance is returned with same handler configuration

    Test type: unit
    """
    logger_name = "test.handler_reuse"

    # First call
    logger1 = get_logger(logger_name, debug=True, output_dir=tmp_path, console_output=False)
    handler_count_1 = len(logger1.handlers)
    handler_ids_1 = [id(h) for h in logger1.handlers]

    # Second call
    logger2 = get_logger(logger_name, debug=True, output_dir=tmp_path, console_output=False)
    handler_count_2 = len(logger2.handlers)
    handler_ids_2 = [id(h) for h in logger2.handlers]

    # Should be the same logger instance
    assert logger1 is logger2, "get_logger should return the same logger instance for the same name"

    # Should have the same handlers
    assert (
        handler_count_1 == handler_count_2
    ), f"Handler count changed from {handler_count_1} to {handler_count_2}"

    # Handler IDs should be the same (handlers were reused, not recreated)
    assert handler_ids_1 == handler_ids_2, "Handlers were recreated instead of being reused"
