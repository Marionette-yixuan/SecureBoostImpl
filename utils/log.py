import logging
import time

logger = logging.getLogger('CART')
logger.propagate = False

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('static/log/log_{}.log'.format(time.strftime('%m-%d %H-%M'), time.localtime()))

logger.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == '__main__':
    logger.info('This is a test log. ')