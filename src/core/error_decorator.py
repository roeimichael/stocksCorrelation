"""Error logging decorator for API functions with detailed stack traces."""

import functools
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def log_errors_to_file(log_file: str = "logs/api_errors.log"):
    """
    Decorator that logs detailed error information to a file when a function fails.

    Logs include:
    - Timestamp of error
    - Function name and module
    - Full stack trace
    - Error type and message
    - Function arguments (with sanitization for sensitive data)

    Args:
        log_file: Path to the log file (default: logs/api_errors.log)

    Usage:
        @log_errors_to_file()
        async def my_api_function(arg1, arg2):
            # function code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log the error
                _log_error_details(func, e, args, kwargs, log_file)
                # Re-raise the exception so it's still handled by FastAPI
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                _log_error_details(func, e, args, kwargs, log_file)
                # Re-raise the exception so it's still handled by FastAPI
                raise

        # Return appropriate wrapper based on whether function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _log_error_details(func: Callable, error: Exception, args: tuple, kwargs: dict, log_file: str):
    """Internal function to format and write error details to log file."""

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Get function details
    func_name = func.__name__
    func_module = func.__module__

    # Get full stack trace
    stack_trace = traceback.format_exc()

    # Format arguments (sanitize sensitive data)
    formatted_args = _format_args(args, kwargs)

    # Create error log entry
    log_entry = f"""
{'='*100}
TIMESTAMP: {timestamp}
FUNCTION:  {func_module}.{func_name}
ERROR TYPE: {type(error).__name__}
ERROR MSG:  {str(error)}

ARGUMENTS:
{formatted_args}

FULL STACK TRACE:
{stack_trace}
{'='*100}

"""

    # Write to log file
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as log_error:
        # If logging fails, at least print to console
        print(f"Failed to write to error log: {log_error}")
        print(log_entry)


def _format_args(args: tuple, kwargs: dict) -> str:
    """Format function arguments for logging, sanitizing sensitive data."""

    formatted = []

    # Format positional args
    if args:
        # Skip 'self' for class methods (first arg)
        start_idx = 0
        if args and hasattr(args[0], '__dict__'):
            start_idx = 1
            formatted.append("  self: <instance>")

        for i, arg in enumerate(args[start_idx:], start=start_idx):
            arg_str = _sanitize_value(arg)
            formatted.append(f"  arg[{i}]: {arg_str}")

    # Format keyword args
    if kwargs:
        for key, value in kwargs.items():
            # Sanitize sensitive keys
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key', 'api']):
                value_str = "***REDACTED***"
            else:
                value_str = _sanitize_value(value)
            formatted.append(f"  {key}: {value_str}")

    return '\n'.join(formatted) if formatted else "  (no arguments)"


def _sanitize_value(value: Any) -> str:
    """Convert a value to a safe string representation."""
    try:
        # Limit string length to avoid huge logs
        max_length = 200

        if isinstance(value, str):
            if len(value) > max_length:
                return f"{value[:max_length]}... (truncated, total length: {len(value)})"
            return repr(value)

        elif isinstance(value, (list, tuple, set)):
            if len(value) > 10:
                return f"{type(value).__name__} with {len(value)} items (first 3: {list(value)[:3]}...)"
            return repr(value)

        elif isinstance(value, dict):
            if len(value) > 10:
                keys = list(value.keys())[:3]
                return f"dict with {len(value)} keys (first 3: {keys}...)"
            return repr(value)

        elif hasattr(value, '__dict__'):
            # For custom objects, show class name and attributes
            return f"<{type(value).__name__} object>"

        else:
            value_str = str(value)
            if len(value_str) > max_length:
                return f"{value_str[:max_length]}... (truncated)"
            return value_str

    except Exception:
        return f"<{type(value).__name__} - could not serialize>"
