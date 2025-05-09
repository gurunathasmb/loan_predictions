import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS_DIR = os.path.join(BASE_DIR, os.getenv('LOGS_DIR'))

# Ensure Logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Define log file path
LOG_FILE = os.path.join(LOGS_DIR, "mlops_training.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger():
    """
    Returns the configured logger.
    """
    return logging.getLogger()

def log_info(message):
    """
    Logs an info-level message.
    """
    logger = get_logger()
    logger.info(message)
    print(f"INFO: {message}")  # Optional console output

def log_error(message):
    """
    Logs an error-level message.
    """
    logger = get_logger()
    logger.error(message)
    print(f"ERROR: {message}")  # Optional console output

def log_warning(message):
    """
    Logs a warning-level message.
    """
    logger = get_logger()
    logger.warning(message)
    print(f"WARNING: {message}")  # Optional console output
