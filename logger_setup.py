import os
import logging
from datetime import datetime

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime('project_%Y%m%d_%H%M%S.log')
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w', encoding='utf-8')]
)
logger = logging.getLogger('project_logger')
