import os
import tempfile
import logging
from pathlib import Path
import random
import numpy as np
from POMDPPlanners.utils.logger import get_logger

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