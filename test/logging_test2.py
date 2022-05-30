import logging
import logging.config
with open("logging.conf") as f: # capture FileNotFoundError error
    logging.config.fileConfig(f) 
# create logger
logger = logging.getLogger('aaasimpleExasmple')
logger.baseFilename = "train.log"
logging.FileHandler
# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

logging.config.dictConfig()