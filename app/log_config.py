import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Specify the directory for logging
    log_directory = 'logs'
    log_filename = 'app.log'
    full_log_path = os.path.join(log_directory, log_filename)

    # Create the directory if it does not exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Configure root logger
    logger = logging.getLogger("machma_logger")
    if not logger.handlers:  # Check if handlers have already been added
        logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(full_log_path, maxBytes=20000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
