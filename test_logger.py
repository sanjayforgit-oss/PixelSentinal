from preprocessing.logger import get_logger

logger = get_logger(__name__)

logger.info("Logger is working!")

logger.warning("This is a warning.")

logger.error("This is an error.")
