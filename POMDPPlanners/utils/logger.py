import logging

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name (str): The name for the logger
        level (int): The logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)
