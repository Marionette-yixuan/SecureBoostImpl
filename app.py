import argparse
from flask import Flask, request
from flask_cors import CORS

from core.PassiveParty import PassiveParty
from utils.log import logger

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/getSampleList', methods=['POST'])
def get_sample_list():
    return pp.get_sample_list()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--id', dest='id', type=int, default=1)
    parser.add_argument('-p', '--port', dest='port', type=int, default=10001)
    args = parser.parse_args()
    
    id = args.id
    port = args.port

    pp = PassiveParty(f'pp{id}')
    pp.load_dataset(f'static/data/pp{id}_train.csv', f'static/data/pp{id}_test.csv')

    local_host = '127.0.0.1'
    app.run(host=local_host, port=port)