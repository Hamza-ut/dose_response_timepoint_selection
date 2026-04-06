import logging
from pathlib import Path
from datetime import datetime


## LOGGING SETUP
# log file name setter
def get_default_log_filename(log_name: str | None = None) -> Path:
    """
    Helper function to setup_logger(), for making log file name with datetimestamp.
    """
    # Get script directory as default log location
    input_dir = Path(__file__).resolve().parent
    # If no log name provided, create one with timestamp
    if not log_name:
        log_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return input_dir / log_name


# Logger setup
def setup_logger(
    name: str = __name__, log_file: str | Path | None = None, log_to_file: bool = False
) -> logging.Logger:
    """
    Configure and return a logger.

    :param name: Logger name.
    :type name: str
    :param log_file: Path to log file.
    :type log_file: str or Path or None
    :param log_to_file: Whether to write logs to file.
    :type log_to_file: bool
    :returns: Configured logger instance.
    :rtype: logging.Logger
    """
    # Default log file name if none provided
    if log_file is None:
        log_file = get_default_log_filename()

    # Create logger (either creates a new logger or returns an existing one with that name.)
    logger = logging.getLogger(name)
    # This logger will handle all messages from DEBUG and above (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    logger.setLevel(logging.DEBUG)

    # long format for console and file (timestamp, __name__, log level, your log message, line number where log was called, filename of the script)
    long_formatter = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s: Line:%(lineno)d: [%(filename)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Shorter format for console output only (timestamp, log level, your log message)
    short_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (always active) to send logs to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(short_formatter)
    # Only INFO and higher are shown in console, DEBUG is silent
    console_handler.setLevel(logging.INFO)

    # Avoid adding multiple handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        if log_file is None:
            log_file = get_default_log_filename()
        file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        file_handler.setFormatter(long_formatter)
        file_handler.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

    return logger
