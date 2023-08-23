import os
import shutil
import argparse
import logging
from flask import Flask, request
from flask_cors import CORS

from core.PassiveParty import PassiveParty
from utils.log import logger

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/initSampleAlign', methods=['POST'])
def init_sample_align():
    return pp.init_sample_align()

@app.route('/getSampleAlign', methods=['POST'])
def get_sample_align():
    return pp.get_sample_align()

@app.route('/recvSampleList', methods=['POST'])
def recv_sample_list():
    recv_data = request.form.to_dict()
    logger.debug(f'Request data: {request.form.keys()}. ')
    return pp.recv_sample_list(recv_data['file_name'])

@app.route('/recvActivePubKey', methods=['POST'])
def recv_active_pub_key():
    recv_data = request.form.to_dict()
    logger.debug(f'Request data: {request.form.keys()}. ')
    return pp.recv_active_pub_key(recv_data['file_name'])

@app.route('/recvGradients', methods=['POST'])
def recv_gradients():
    recv_data = request.form.to_dict()
    logger.debug(f'Request data: {request.form.keys()}. ')
    return pp.recv_gradients(recv_data)

@app.route('/getSplitsSum', methods=['POST'])
def get_splits_sum():
    return pp.get_splits_sum()

@app.route('/confirmSplit', methods=['POST'])
def confirm_split():
    recv_data = request.form.to_dict()
    logger.debug(f'Request data: {request.form.keys()}. ')
    return pp.confirm_split(recv_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--id', dest='id', type=int, default=1)
    parser.add_argument('-p', '--port', dest='port', type=int, default=10001)
    args = parser.parse_args()
    
    id = args.id
    port = args.port

    path_list = [
        f'temp/file/party-{id}', 
        f'temp/model/party-{id}'
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    pp = PassiveParty(id)
    pp.load_dataset(f'static/data/pp{id}_train.csv', f'static/data/pp{id}_test.csv')

    local_host = '127.0.0.1'
    app.run(host=local_host, port=port)