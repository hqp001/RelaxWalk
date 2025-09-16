"""
Centralized logging configuration for RelaxWalk algorithms.

This module provides a unified logging setup for tracking runtime performance,
algorithm progress, and debugging information across the entire codebase.
"""

import logging
import sys
import time
import inspect
from pathlib import Path
from typing import Optional, Any, Dict
from functools import wraps


class PerformanceLogger:
    """
    Enhanced logger for tracking algorithm performance and runtime metrics.
    """

    def __init__(self, name: str = "RelaxWalk", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Setup console and file handlers with proper formatting."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / f"relaxwalk_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _get_caller_info(self, skip_frames=2):
        """Get information about the calling function."""
        frame = inspect.currentframe()
        try:
            # Skip frames: _get_caller_info -> info/debug/warning/error -> actual caller
            for _ in range(skip_frames):
                frame = frame.f_back
                if frame is None:
                    return "unknown"
            return frame.f_code.co_name
        finally:
            del frame

    def info(self, msg: str, **kwargs):
        """Log info message with optional extra data."""
        extra_str = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        caller = self._get_caller_info()
        full_msg = f"[{caller}] {msg} {extra_str}".strip()
        self.logger.info(full_msg)

    def debug(self, msg: str, **kwargs):
        """Log debug message with optional extra data."""
        extra_str = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        caller = self._get_caller_info()
        full_msg = f"[{caller}] {msg} {extra_str}".strip()
        self.logger.debug(full_msg)

    def warning(self, msg: str, **kwargs):
        """Log warning message with optional extra data."""
        extra_str = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        caller = self._get_caller_info()
        full_msg = f"[{caller}] {msg} {extra_str}".strip()
        self.logger.warning(full_msg)

    def error(self, msg: str, **kwargs):
        """Log error message with optional extra data."""
        extra_str = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        caller = self._get_caller_info()
        full_msg = f"[{caller}] {msg} {extra_str}".strip()
        self.logger.error(full_msg)


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, logger: PerformanceLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None

    def _get_caller_function(self):
        """Get the function name that created this context manager."""
        frame = inspect.currentframe()
        try:
            # Skip: _get_caller_function -> __enter__/__exit__ -> with statement -> actual function
            for _ in range(4):
                frame = frame.f_back
                if frame is None:
                    return "unknown"
            return frame.f_code.co_name
        finally:
            del frame

    def __enter__(self):
        self.start_time = time.time()
        context_str = " | ".join(f"{k}={v}" for k, v in self.context.items()) if self.context else ""
        msg = f"Starting {self.operation}"
        if context_str:
            msg += f" | {context_str}"
        caller = self._get_caller_function()
        full_msg = f"[{caller}] {msg}"
        self.logger.debug(full_msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        context_str = " | ".join(f"{k}={v}" for k, v in self.context.items()) if self.context else ""
        msg = f"Completed {self.operation}"
        if context_str:
            msg += f" | {context_str}"
        msg += f" | duration={duration:.3f}s"

        caller = self._get_caller_function()

        if exc_type is None:
            full_msg = f"[{caller}] {msg}"
            self.logger.info(full_msg)
        else:
            full_msg = f"[{caller}] Failed {self.operation} | duration={duration:.3f}s | error={exc_val}"
            self.logger.error(full_msg)


def timer(operation_name: Optional[str] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            op_name = operation_name or f"{func.__name__}()"

            with TimingContext(logger, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_logger(name: str = "RelaxWalk", log_level: str = "INFO") -> PerformanceLogger:
    """
    Get or create a performance logger instance.

    Args:
        name: Logger name (typically module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        PerformanceLogger instance
    """
    return PerformanceLogger(name, log_level)


# Global logger instance for convenience
logger = get_logger()