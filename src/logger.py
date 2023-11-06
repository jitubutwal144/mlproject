import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_PATH = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOGS_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

logging.basicConfig(
  filename=LOG_FILE_PATH,
  format="[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(message)s",
  level=logging.INFO
)

# if __name__ == '__main__':
#   logging.info('Testing logger')