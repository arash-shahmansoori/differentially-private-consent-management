import logging


def get_logger():
    """Create logger and set the logging level"""

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Formatting
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    # File handling
    file_handler = logging.FileHandler("training.log")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
