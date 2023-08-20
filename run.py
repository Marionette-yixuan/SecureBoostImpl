import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from core.ActiveParty import ActiveParty


def create_passive_party(params):
    id = params['id']
    port = params['port']
    os.system(f'python app.py -i {id} -p {port}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--number', dest='passive_num', type=int, default=2)
    args = parser.parse_args()

    passive_num = args.passive_num
    start_port = 10001

    executor = ThreadPoolExecutor(10)

    passive_list = []
    for i in range(passive_num):
        port = start_port + i
        passive_list.append(port)
        executor.submit(create_passive_party, ({'id': i + 1, 'port': port}))

    ap = ActiveParty('ap', passive_list)
    ap.load_dataset('static/data/ap_train.csv', 'static/data/ap_test.csv')
    # ap.broadcast_pub_key()
    ap.sample_align()