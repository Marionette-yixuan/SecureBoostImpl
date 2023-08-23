from concurrent.futures import ThreadPoolExecutor
from utils.params import passive_list


def broadcast(func):
    """
    开启线程对所有被动方端口进行广播。
    func 方法的第一个参数必须为 port
    """
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(10) as executor:
            for port in passive_list:
                executor.submit(func, port, *args, **kwargs))
    return wrapper

def use_thread(func):
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(10) as executor:
            executor.submit(func, *args, **kwargs)
    return wrapper