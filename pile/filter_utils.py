import datasets
import logging
import rich 
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler())


logger.info(f"Dataset is here")