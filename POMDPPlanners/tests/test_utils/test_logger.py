import os
import tempfile
import logging
from pathlib import Path
from POMDPPlanners.utils.logger import get_logger

def test_logger_console_output_true(capsys):
    """Test logger console output true.
    
    Purpose: Validates logger console output true
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    logger = get_logger("test.console", debug=True, console_output=True)
    logger.info("This is a test message for console output.")
    captured = capsys.readouterr()
    assert "This is a test message for console output." in captured.out or captured.err


def test_logger_console_output_false(capsys):
    """Test logger console output false.
    
    Purpose: Validates logger console output false
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    logger = get_logger("test.no_console", debug=True, console_output=False)
    logger.info("This should not appear in the console.")
    captured = capsys.readouterr()
    assert "This should not appear in the console." not in captured.out
    assert "This should not appear in the console." not in captured.err


def test_logger_file_output(tmp_path):
    """Test logger file output.
    
    Purpose: Validates logger file output
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
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
    
    Purpose: Validates logger no duplicate handlers
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
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