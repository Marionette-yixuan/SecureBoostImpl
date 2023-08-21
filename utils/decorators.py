import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from utils.params import passive_list


def broadcast(func):
    """
    开启线程对所有被动方端口进行广播，在所有被动方返回后结束。
    func 方法的第一个参数必须为 port
    """
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(10) as executor:
            futures = []
            for port in passive_list:
                futures.append(executor.submit(func, port, *args, **kwargs))
            concurrent.futures.wait(futures)
    return wrapper