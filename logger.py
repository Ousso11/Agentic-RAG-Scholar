import logging
from termcolor import colored


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "blue",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "magenta",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "white")
        record.msg = colored(record.msg, color)
        return super().format(record)


class Logger:
    def __init__(self, name=__name__, level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


# Usage
# logger = Logger(__name__).get_logger()

# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")
