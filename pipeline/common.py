import os
import logging
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_dir=Path(__file__).resolve().parent.parent / "logs", subdirectory=''):
    """Set up Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    datetime_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    full_log_dir = os.path.join(log_dir, subdirectory, datetime_folder)
    os.makedirs(full_log_dir, exist_ok=True)

    log_filename = f"{name}.log"
    
    file_handler = logging.FileHandler(os.path.join(full_log_dir, log_filename))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger