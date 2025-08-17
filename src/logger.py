import logging
import os
from datetime import datetime

# Only create logs directory if we're actually going to use it
# For now, disable file logging to prevent empty files
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Disable file logging to prevent empty log files
# Uncomment the lines below if you actually need file logging
# os.makedirs(logs_path, exist_ok=True)
# LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure console logging only (no file logging)
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Console output only
    ]
)

# If you need file logging later, uncomment this:
# file_handler = logging.FileHandler(LOG_FILE_PATH)
# file_handler.setFormatter(logging.Formatter(
#     "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
# ))
# logging.getLogger().addHandler(file_handler)
