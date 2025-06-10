import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, output_dir: Optional[Path] = None, debug: bool = False) -> logging.Logger:
    """
    Get a configured logger instance with optional file and console output.
    
    Args:
        name (str): The name for the logger
        level (int): The logging level (default: logging.INFO)
        output_dir (Optional[Path]): Directory to store log files. If None, only console logging is enabled.
        debug (bool): Whether to enable debug logging (default: False)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if output_dir is provided
    if output_dir is not None:
        # Convert string to Path if needed
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        # Create logs directory
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use logger name (which includes policy name) in the filename
        name_part = name.replace('.', '_')  # Replace dots with underscores for filename
        log_file = logs_dir / f"{name_part}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if debug else level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger
