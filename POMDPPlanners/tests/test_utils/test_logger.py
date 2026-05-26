# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

import logging
import random
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from POMDPPlanners.utils.logger import (
    cleanup_all_loggers,
    cleanup_task_logger,
    ConditionalMemoryHandler,
    flush_buffered_task_logs,
    get_logger,
    get_queue_logger_diagnostics,
    get_task_logger_manager,
    setup_task_logger_with_buffering,
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


def test_logger_no_handlers_no_io(capsys, tmp_path):
    """Test logger with no handlers performs no I/O.

    Purpose: Validates that logger with console_output=False and output_dir=None has zero handlers
             and performs no I/O operations

    Given: Logger configured with console_output=False and output_dir=None
    When: Multiple messages are logged at various levels
    Then: Logger has zero handlers (or only NullHandlers which perform no I/O), no console output, and no files are created

    Test type: unit
    """
    logger = get_logger(name="test.no_handlers", debug=True, console_output=False, output_dir=None)

    # Verify logger has zero handlers OR only NullHandlers (which perform no I/O)
    # NullHandler is added to prevent Python's lastResort handler from outputting to stderr
    non_null_handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert (
        len(non_null_handlers) == 0
    ), f"Expected 0 non-NullHandler handlers, got {len(non_null_handlers)}"

    # Log messages at various levels - should not crash
    logger.debug("Debug message - should be ignored")
    logger.info("Info message - should be ignored")
    logger.warning("Warning message - should be ignored")
    logger.error("Error message - should be ignored")

    # Verify no console output
    captured = capsys.readouterr()
    assert "Debug message" not in captured.out
    assert "Info message" not in captured.out
    assert "Warning message" not in captured.out
    assert "Error message" not in captured.out
    assert "Debug message" not in captured.err
    assert "Info message" not in captured.err
    assert "Warning message" not in captured.err
    assert "Error message" not in captured.err

    # Verify no log files were created
    logs_dir = tmp_path / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        assert len(log_files) == 0, f"Expected no log files, found {len(log_files)}"


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


def test_conditional_memory_handler_buffering(tmp_path):
    """Test that ConditionalMemoryHandler buffers logs in memory.

    Purpose: Validates that ConditionalMemoryHandler holds logs in buffer without writing to disk

    Given: A ConditionalMemoryHandler wrapping a FileHandler
    When: INFO level messages are logged
    Then: Messages are buffered in memory and not written to the file

    Test type: unit
    """
    log_file = tmp_path / "test_buffer.log"

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create memory handler wrapping file handler
    memory_handler = ConditionalMemoryHandler(capacity=100, target=file_handler)
    memory_handler.setLevel(logging.INFO)

    # Create logger with memory handler
    logger = logging.getLogger("test.buffer")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(memory_handler)

    # Log some messages
    logger.info("Buffered message 1")
    logger.info("Buffered message 2")

    # File should be empty (logs are buffered)
    if log_file.exists():
        assert log_file.stat().st_size == 0, "File should be empty before flush"

    # Buffer should have messages
    assert len(memory_handler.buffer) == 2


def test_conditional_memory_handler_flush_on_error(tmp_path):
    """Test that ConditionalMemoryHandler auto-flushes on ERROR level.

    Purpose: Validates that ERROR level messages trigger automatic flush of buffered logs

    Given: A ConditionalMemoryHandler with buffered INFO messages
    When: An ERROR level message is logged
    Then: All buffered messages (including ERROR) are flushed to file

    Test type: unit
    """
    log_file = tmp_path / "test_error_flush.log"

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Create memory handler wrapping file handler
    memory_handler = ConditionalMemoryHandler(capacity=100, target=file_handler)
    memory_handler.setLevel(logging.INFO)
    memory_handler.setFormatter(formatter)

    # Create logger with memory handler
    logger = logging.getLogger("test.error_flush")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(memory_handler)

    # Log INFO messages (should be buffered)
    logger.info("Buffered message 1")
    logger.info("Buffered message 2")

    # Log ERROR message (should trigger flush)
    logger.error("Error message - triggers flush")

    # File should now contain all messages
    assert log_file.exists()
    with open(log_file, "r") as f:
        content = f.read()

    assert "Buffered message 1" in content
    assert "Buffered message 2" in content
    assert "Error message - triggers flush" in content


def test_conditional_memory_handler_manual_flush(tmp_path):
    """Test that ConditionalMemoryHandler.trigger_flush() works correctly.

    Purpose: Validates that manual flush trigger writes buffered logs to file

    Given: A ConditionalMemoryHandler with buffered INFO messages
    When: trigger_flush() is called manually
    Then: All buffered messages are flushed to file

    Test type: unit
    """
    log_file = tmp_path / "test_manual_flush.log"

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    # Create memory handler wrapping file handler
    memory_handler = ConditionalMemoryHandler(capacity=100, target=file_handler)
    memory_handler.setLevel(logging.INFO)
    memory_handler.setFormatter(formatter)

    # Create logger with memory handler
    logger = logging.getLogger("test.manual_flush")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(memory_handler)

    # Log messages (should be buffered)
    logger.info("Message 1")
    logger.info("Message 2")
    logger.info("Message 3")

    # Manually trigger flush
    memory_handler.trigger_flush()

    # File should now contain all messages
    assert log_file.exists()
    with open(log_file, "r") as f:
        content = f.read()

    assert "Message 1" in content
    assert "Message 2" in content
    assert "Message 3" in content


def test_setup_task_logger_with_buffering_creates_logger(tmp_path):
    """Test that setup_task_logger_with_buffering creates a properly configured logger.

    Purpose: Validates that setup function creates logger with correct configuration

    Given: Logger name and configuration parameters
    When: setup_task_logger_with_buffering is called with log_only_on_failure=False
    Then: Logger is created with file and console handlers as expected

    Test type: unit
    """
    logger_name = "test.task.logger.create"

    logger = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=False,
    )

    assert logger is not None
    assert logger.name == logger_name
    assert len(logger.handlers) > 0

    # Check that log file is created
    logger.info("Test message")
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0


def test_setup_task_logger_with_buffering_adds_memory_handlers(tmp_path):
    """Test that setup_task_logger_with_buffering adds ConditionalMemoryHandler when enabled.

    Purpose: Validates that log_only_on_failure=True wraps handlers with ConditionalMemoryHandler

    Given: Logger configuration with log_only_on_failure=True
    When: setup_task_logger_with_buffering is called
    Then: Handlers are wrapped with ConditionalMemoryHandler for buffering

    Test type: unit
    """
    logger_name = "test.task.logger.buffering"

    logger = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=True,
    )

    # Check that handlers are ConditionalMemoryHandler
    has_memory_handler = any(isinstance(h, ConditionalMemoryHandler) for h in logger.handlers)
    assert has_memory_handler, "Should have ConditionalMemoryHandler when log_only_on_failure=True"

    # Verify messages are buffered
    logger.info("Buffered test message")

    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("*.log"))

    # File should exist but be empty (buffered)
    if log_files:
        for log_file in log_files:
            assert log_file.stat().st_size == 0, "Log file should be empty (logs buffered)"


def test_setup_task_logger_with_buffering_reuses_existing(tmp_path):
    """Test that setup_task_logger_with_buffering reuses existing configured logger.

    Purpose: Validates that repeated calls with same logger name return the same logger

    Given: Logger already created and configured
    When: setup_task_logger_with_buffering is called again with same name
    Then: Same logger instance is returned without duplicate handlers

    Test type: unit
    """
    logger_name = "test.task.logger.reuse"

    # First call
    logger1 = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=False,
    )
    handler_count_1 = len(logger1.handlers)

    # Second call
    logger2 = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=False,
    )
    handler_count_2 = len(logger2.handlers)

    # Should be the same logger instance
    assert logger1 is logger2
    # Should not have added duplicate handlers
    assert handler_count_1 == handler_count_2


def test_flush_buffered_task_logs_flushes_handlers(tmp_path):
    """Test that flush_buffered_task_logs flushes buffered logs to file.

    Purpose: Validates that flush function triggers flush of all buffered logs

    Given: Logger with buffered messages (log_only_on_failure=True)
    When: flush_buffered_task_logs is called
    Then: Buffered logs are written to file

    Test type: unit
    """
    logger_name = "test.task.flush"

    logger = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=True,
    )

    # Log messages (should be buffered)
    logger.info("Buffered message 1")
    logger.info("Buffered message 2")

    # Flush buffered logs
    flush_buffered_task_logs(logger_name)

    # Check that logs were written to file
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0

    # Read log file
    with open(log_files[0], "r") as f:
        content = f.read()

    assert "Buffered message 1" in content
    assert "Buffered message 2" in content


def test_cleanup_task_logger_success_discards_logs(tmp_path):
    """Test that cleanup_task_logger discards buffered logs on success.

    Purpose: Validates that successful episodes with log_only_on_failure discard buffered logs

    Given: Logger with buffered messages and log_only_on_failure=True
    When: cleanup_task_logger is called with episode_failed=False
    Then: Buffered logs are discarded without writing to file

    Test type: unit
    """
    logger_name = "test.task.cleanup.success"

    logger = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=True,
    )

    # Log messages (should be buffered)
    logger.info("Should be discarded 1")
    logger.info("Should be discarded 2")

    # Cleanup with success (episode_failed=False)
    cleanup_task_logger(
        logger_name=logger_name,
        episode_failed=False,
        log_only_on_failure=True,
    )

    # Check that log file is empty or doesn't have the messages
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("*.log"))

    if log_files:
        for log_file in log_files:
            # File should be empty (logs discarded)
            assert log_file.stat().st_size == 0, "Log file should be empty after cleanup on success"


def test_cleanup_task_logger_failure_flushes_logs(tmp_path):
    """Test that cleanup_task_logger flushes buffered logs on failure.

    Purpose: Validates that failed episodes with log_only_on_failure flush buffered logs

    Given: Logger with buffered messages and log_only_on_failure=True
    When: cleanup_task_logger is called with episode_failed=True
    Then: Buffered logs are flushed and written to file

    Test type: unit
    """
    logger_name = "test.task.cleanup.failure"

    logger = setup_task_logger_with_buffering(
        logger_name=logger_name,
        output_dir=tmp_path,
        debug=True,
        console_output=False,
        use_queue=False,
        log_only_on_failure=True,
    )

    # Log messages (should be buffered)
    logger.info("Failure message 1")
    logger.info("Failure message 2")

    # Cleanup with failure (episode_failed=True)
    cleanup_task_logger(
        logger_name=logger_name,
        episode_failed=True,
        log_only_on_failure=True,
    )

    # Check that logs were written to file
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0

    # Read log file
    with open(log_files[0], "r") as f:
        content = f.read()

    assert "Failure message 1" in content
    assert "Failure message 2" in content


def test_task_logger_manager_thread_safety(tmp_path):
    """Test that TaskLoggerManager is thread-safe for concurrent access.

    Purpose: Validates that TaskLoggerManager can handle concurrent logger creation safely

    Given: Multiple threads creating loggers simultaneously
    When: Threads call setup_task_logger_with_buffering concurrently
    Then: All loggers are created correctly without race conditions

    Test type: unit
    """
    num_threads = 10
    results = []
    errors = []

    def create_logger(thread_id):
        try:
            logger_name = f"test.task.thread.{thread_id}"
            logger = setup_task_logger_with_buffering(
                logger_name=logger_name,
                output_dir=tmp_path,
                debug=False,
                console_output=False,
                use_queue=False,
                log_only_on_failure=False,
            )
            logger.info(f"Message from thread {thread_id}")
            results.append((thread_id, logger.name))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Create threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=create_logger, args=(i,))
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify all threads created loggers
    assert len(results) == num_threads

    # Verify log files were created
    log_dir = tmp_path / "logs"
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == num_threads


def test_task_logger_manager_state_tracking(tmp_path):
    """Test that TaskLoggerManager correctly tracks logger state.

    Purpose: Validates that TaskLoggerManager maintains accurate state of configured loggers

    Given: TaskLoggerManager with multiple loggers created
    When: Loggers are created and configured
    Then: Manager correctly tracks which loggers are configured and their handlers

    Test type: unit
    """
    manager = get_task_logger_manager()

    # Clear any previous state
    manager._configured_loggers.clear()
    manager._memory_handlers.clear()

    logger_name_1 = "test.state.tracking.1"
    logger_name_2 = "test.state.tracking.2"

    # Create first logger without buffering
    logger1 = manager.get_or_create_logger(
        logger_name=logger_name_1,
        output_dir=tmp_path,
        debug=False,
        console_output=False,
        use_queue=False,
        log_only_on_failure=False,
    )

    # Check state
    assert logger_name_1 in manager._configured_loggers
    assert logger_name_1 not in manager._memory_handlers  # No buffering

    # Create second logger with buffering
    logger2 = manager.get_or_create_logger(
        logger_name=logger_name_2,
        output_dir=tmp_path,
        debug=False,
        console_output=False,
        use_queue=False,
        log_only_on_failure=True,
    )

    # Check state
    assert logger_name_2 in manager._configured_loggers
    assert logger_name_2 in manager._memory_handlers  # Has buffering
    assert len(manager._memory_handlers[logger_name_2]) > 0

    # Verify loggers are different
    assert logger1 is not logger2
