import logging
import time
import os
from datetime import datetime, timedelta

log_path = 'static/log'
if not os.path.exists(log_path):
    os.makedirs(log_path)

logger = logging.getLogger('vfl')
logger.propagate = False

stream_handler = logging.StreamHandler()
debug_handler = logging.FileHandler(
    os.path.join(log_path, 'log_{}.log'.format(time.strftime('%m-%d %H-%M'), time.localtime())))
info_handler = logging.FileHandler(
    os.path.join(log_path, 'log_{}_csc.log'.format(time.strftime('%m-%d %H-%M'), time.localtime()))
)


logger.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
info_handler.setLevel(logging.INFO)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')
debug_formatter = logging.Formatter('[%(levelname)s] File "%(module)s/%(filename)s", line %(lineno)d > %(message)s')
stream_handler.setFormatter(formatter)
debug_handler.setFormatter(debug_formatter)
info_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(debug_handler)
logger.addHandler(info_handler)


def delete_previous_log():
    """
    删除之前的日志
    """
    today = datetime.now().date()
    previous_date = today - timedelta(days=1)
    previous_date = previous_date.strftime('%m-%d')
    for file_name in os.listdir(log_path):
        if file_name.startswith(f'log_{previous_date}'):
            os.remove(os.path.join(log_path, file_name))


delete_previous_log()


if __name__ == '__main__':
    logger.info('This is a test log. ')
    logger.debug('This is a test debug log. ')
    