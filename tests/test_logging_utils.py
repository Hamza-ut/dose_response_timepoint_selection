import logging
from drc_timepoint.logging_utils import setup_logger


def test_setup_logger_file_creation(tmp_path):
    # 1. SETUP: Point the logger to a temp file
    log_file = tmp_path / "test.log"

    # 2. ACTION
    logger = setup_logger(name="test_logger", log_file=log_file, log_to_file=True)
    logger.info("Test message")

    # 3. ASSERT
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content


def test_setup_logger_no_duplicate_handlers():
    # 1. ACTION: Setup the same logger twice
    logger = setup_logger(name="duplicate_test")
    logger = setup_logger(name="duplicate_test")

    # 2. ASSERT: Ensure only one console handler exists
    console_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(console_handlers) == 1
