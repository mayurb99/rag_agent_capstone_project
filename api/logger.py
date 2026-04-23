import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("medical_agent")


def log(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat()
    logger.info(json.dumps(data))
