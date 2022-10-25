from structlog import wrap_logger
from structlog.stdlib import filter_by_level
import logging
import datetime
from typing import Union
import sys


def get_logger(name:str=__name__, loglevel: Union[str, int] = "INFO"):
	logging.basicConfig(stream=sys.stdout, format= "%(name)s — %(message)s", level=loglevel)
	log = wrap_logger(logging.getLogger(name))
	return log