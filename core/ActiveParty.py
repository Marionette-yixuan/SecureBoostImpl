import os
import json
import threading
import requests
import pandas as pd
from collections import deque
from phe import paillier
from concurrent.futures import ThreadPoolExecutor

from utils.log import logger
from utils.params import temp_root
from msgs.messages import msg_empty, msg_name_file

class ActiveParty:
    def __init__(self, passive_list: list) -> None:
        self.id = 0
        self.name = f'party{self.id}'
        self.pub_key, self.pri_key = paillier.generate_paillier_keypair(n_length=256)
        logger.info(f'{self.name.upper()}: Paillier key generated. ')

        self.dataset = None
        self.validset = None

        self.model = Model()

        self.passive_list = passive_list        # 所有被动方的端口

        # 训练中临时变量
        self.cur_split_node = None      # 当前正在分裂的节点
        self.split_nodes = deque()      # 待分裂的节点队列
        self.feature_split_time = None  # 各特征的分裂次数

    def load_dataset(self, data_path: str, valid_path: str=None):
        """
        加载数据集
        """
        dataset = pd.read_csv(data_path)
        dataset['id'] = dataset['id'].astype(str)
        self.dataset = dataset.set_index('id')

        if valid_path:
            validset = pd.read_csv(valid_path)
            validset['id'] = validset['id'].astype(str)
            self.validset = validset.set_index('id')

        self.feature_split_time = pd.Series(0, index=self.dataset.index)

    def sample_align(self):
        """
        样本对齐
        """
        lock = threading.Lock()

        from utils.sha1 import sha1

        train_idx = self.dataset.index.tolist()
        train_idx_map = {sha1(idx): idx for idx in train_idx}
        train_hash = set(train_idx_map.keys())

        if self.validset is not None:
            valid_idx = self.validset.index.tolist()
            valid_idx_map = {sha1(idx): idx for idx in valid_idx}
            valid_hash = set(valid_idx_map.keys())

        
        # 向所有被动方请求样本列表，返回时依次求交
        count = 0
        def recv_sample_list(port: int):
            nonlocal train_hash, valid_hash, count

            data = msg_empty()
            res_dict = requests.post(f'http://127.0.0.1:{port}/getSampleList', data=data).json()
            with open(res_dict['file_name'], 'r') as f:
                hash_data = json.load(f)
            logger.info(f'{self.name.upper()}: Received sample list from {res_dict["party_name"]}')
            
            lock.acquire()          # 加锁防止完整性破坏
            train_hash = train_hash.intersection(set(hash_data['train_hash']))
            if self.validset is not None:
                valid_hash = valid_hash.intersection(set(hash_data['valid_hash']))
            lock.release()
            count += 1

        executor = ThreadPoolExecutor(10)
        for port in self.passive_list:
            executor.submit(recv_sample_list, port)
        while count < len(self.passive_list):       # 同步，接收所有被动方的样本后再进行下一步
            pass
        
        logger.info(f'{self.name.upper()}: Sample alignment finished. Intersect trainset contains {len(train_hash)} samples. ')

        train_hash, valid_hash = list(train_hash), list(valid_hash)     # 将求交后的哈希集合转换为列表

        json_data = {
            'train_hash': train_hash, 
            'valid_hash': valid_hash
        }

        file_name = os.path.join(temp_root['file'][self.id], f'sample_align.json')
        with open(file_name, 'w+') as f:
            json.dump(json_data, f)

        # 向所有被动方广播求交后的样本下标
        count = 0
        def send_aligned_sample(file_name: str, port: int):
            nonlocal count
            data = msg_name_file(self.name, file_name)
            logger.info(f'Sending aligned sample: {data}.')
            requests.post(f'http://127.0.0.1:{port}/recvSampleList', data=data)
            count += 1

        for port in self.passive_list:
            executor.submit(send_aligned_sample, file_name, port)
        while count < len(self.passive_list):
            pass

        logger.info(f'{self.name.upper()}: Aligned sample broadcasted to all passive parties. ')

        self.dataset = self.dataset.loc[[train_idx_map[th] for th in train_hash], :]
        if self.validset is not None:
            self.validset = self.validset.loc[[valid_idx_map[vh] for vh in valid_hash], :]
        

class Model:
    def __init__(self) -> None:
        pass


class TreeNode:
    def __init__(self, id: int, instance_space: list) -> None:
        self.id = id                            # 节点标号
        self.instance_space = instance_space    # 节点包含的样本空间（只在训练和预测过程中有用，不会存储）

        # 分裂信息
        self.party_name = None                  # 该节点所属的训练方
        self.feature_split = None               # 当 self.party_name 为主动方时生效，记录分裂的特征名称和门限值。格式为 {'feature_name': xxx, 'feature_thresh': xxx}
        self.look_up_id = 0                     # 当 self.party_name 为被动方时生效，记录该节点的分裂方式存储在被动方的查找表第几行中

        # 左右子树
        self.left = None                        # 样本对应特征 >= 门限值时进入左子树
        self.right = None

        # 节点权重
        self.weight = 0.0