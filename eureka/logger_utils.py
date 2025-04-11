import logging
from datetime import datetime
from pathlib import Path

def init_logging(log_dir="logs", prefix="eureka_run", level=logging.DEBUG):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{prefix}_{timestamp}.log"

    logging.basicConfig(
        filename=str(log_file),
        filemode='w',
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )

    logging.info(f"Log File: {log_file}")