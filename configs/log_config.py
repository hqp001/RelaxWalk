# ADDED: Centralized logging configuration
import logging
import sys
from datetime import datetime

def setup_logging(log_level=logging.CRITICAL, log_file=None):
    """
    Setup centralized logging configuration for RelaxWalk.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path (default: logs/relaxation_walk_YYYYMMDD.log)
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f'logs/relaxation_walk_{timestamp}.log'

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file

    # Console handler - show all levels including DEBUG
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)  # Show DEBUG logs on console too

    # Configure root logger to DEBUG level to capture everything
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    return logging.getLogger(__name__)

# Initialize logging when module is imported
logger = setup_logging()
logger.info("Logging configuration initialized")

# Export logger for use in other modules
def get_logger(name=None):
    """Get a logger instance for a specific module."""
    return logging.getLogger(name or __name__)
