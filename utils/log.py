import logging
import time

import os
log_path = 'static/log'
if not os.path.exists(log_path):
    os.makedirs(log_path)

logger = logging.getLogger('CART')
logger.propagate = False

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    os.path.join(log_path, 'log_{}.log'.format(time.strftime('%m-%d %H-%M'), time.localtime())))

logger.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')
debug_formatter = logging.Formatter('[%(levelname)s] File "%(module)s/%(filename)s", line %(lineno)d > %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(debug_formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == '__main__':
    logger.info('This is a test log. ')
    logger.debug('This is a test debug log. ')
    