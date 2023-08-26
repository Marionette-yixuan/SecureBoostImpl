import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

from core.ActiveParty import ActiveParty
from utils.params import passive_list


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

    id = 0
    path_list = [
        f'temp/file/party-{id}', 
        f'temp/model/party-{id}'
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    for i in range(passive_num):
        port = start_port + i
        passive_list.append(port)
        executor.submit(create_passive_party, {'id': i + 1, 'port': port})

    ap = ActiveParty()
    ap.load_dataset('static/data/ap_train.csv', 'static/data/ap_test.csv')
    ap.train()
    # file_name = ap.dump_model('static/model/')
    # ap.load_model(file_name)
    ap.predict()
    