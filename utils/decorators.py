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
                executor.submit(func, port, *args, **kwargs)
    return wrapper

def use_thread(func):
    """
    将方法用线程调用
    """
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(10) as executor:
            executor.submit(func, *args, **kwargs)
    return wrapper

def poll(func):
    """
    轮询访问每个被动方，并根据返回数据包的键判断是否返回有效值
    func 方法的第一个参数必须为 port, 返回值必须为 True / False
    """
    def wrapper(*args, **kwargs):
        check_dict = {port: False for port in passive_list}
        with ThreadPoolExecutor(10) as executor:
            while not all(checked for checked in check_dict.values()):
                for port in check_dict:
                    if check_dict[port]:
                        continue
                    future = executor.submit(func, port, *args, **kwargs)
                    if future.result():
                        check_dict[port] = True
    return wrapper