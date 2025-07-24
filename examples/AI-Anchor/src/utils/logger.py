"""
Logging configuration for AI Anchor system.

Provides structured logging with different levels and output formats.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
    from loguru._defaults import LOGURU_FORMAT
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    logger = None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_json: bool = False,
    enable_colors: bool = True,
) -> None:
    """
    Set up logging configuration for the AI Anchor system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_json: Enable JSON format logging
        enable_colors: Enable colored console output
    """
    if not LOGURU_AVAILABLE:
        # Fallback to standard logging
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        print(f"Logging initialized at level: {level} (using standard logging)")
        return
    
    # Remove default logger
    logger.remove()
    
    # Console handler
    if enable_colors:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        format_string = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=enable_colors,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if enable_json:
            logger.add(
                log_file,
                format="{time} | {level} | {name}:{function}:{line} | {message}",
                level=level,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                serialize=True,  # JSON format
            )
        else:
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                level=level,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
            )
    
    # Set up intercept handler for standard logging
    setup_standard_logging_intercept()
    
    logger.info(f"Logging initialized at level: {level}")


def setup_standard_logging_intercept():
    """
    Intercept standard Python logging and route to loguru.
    
    This ensures that all libraries using standard logging
    (like FastAPI, uvicorn, etc.) use our loguru configuration.
    """
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Remove all existing handlers
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger().setLevel(0)

    # Specific loggers that should use our configuration
    for logger_name in [
        "uvicorn",
        "uvicorn.error", 
        "uvicorn.access",
        "fastapi",
        "transformers",
        "torch",
        "vllm",
        "ollama",
    ]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        logging.getLogger(logger_name).setLevel(0)


def get_logger(name: str):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance (loguru or standard logging)
    """
    if LOGURU_AVAILABLE:
        return logger.bind(name=name)
    else:
        return logging.getLogger(name)


# Performance logging utilities
def log_performance(operation: str):
    """
    Decorator for logging performance metrics.
    
    Args:
        operation: Name of the operation being measured
    """
    def decorator(func):
        import time
        import asyncio
        from functools import wraps
        
        perf_logger = get_logger("performance")
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    if LOGURU_AVAILABLE:
                        perf_logger.info(f"⏱️ {operation} completed in {duration:.3f}s")
                    else:
                        perf_logger.info(f"{operation} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    if LOGURU_AVAILABLE:
                        perf_logger.error(f"❌ {operation} failed after {duration:.3f}s: {e}")
                    else:
                        perf_logger.error(f"{operation} failed after {duration:.3f}s: {e}")
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    if LOGURU_AVAILABLE:
                        perf_logger.info(f"⏱️ {operation} completed in {duration:.3f}s")
                    else:
                        perf_logger.info(f"{operation} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    if LOGURU_AVAILABLE:
                        perf_logger.error(f"❌ {operation} failed after {duration:.3f}s: {e}")
                    else:
                        perf_logger.error(f"{operation} failed after {duration:.3f}s: {e}")
                    raise
            return sync_wrapper
    return decorator


def log_gpu_memory(operation: str = "GPU Memory"):
    """
    Log current GPU memory usage.
    
    Args:
        operation: Description of the operation
    """
    gpu_logger = get_logger("gpu_memory")
    
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            if LOGURU_AVAILABLE:
                gpu_logger.info(f"🔧 {operation} - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
            else:
                gpu_logger.info(f"{operation} - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        else:
            gpu_logger.warning("CUDA not available for memory logging")
    except ImportError:
        gpu_logger.warning("PyTorch not available for GPU memory logging")