import os
import json
import threading
import requests
import pandas as pd
from collections import deque
from phe import paillier, PaillierPublicKey, PaillierPrivateKey

from core.Calculator import Calculator
from utils.log import logger
from utils.params import temp_root
from utils.encryption import serialize_pub_key, serialize_encrypted_number, load_encrypted_number
from utils.decorators import broadcast, poll
from msgs.messages import msg_empty, msg_name_file, msg_gradient_file


class ActiveParty:
    def __init__(self) -> None:
        self.id = 0
        self.name = f'party{self.id}'
        self.pub_key, self.pri_key = paillier.generate_paillier_keypair(n_length=Calculator.keypair_gen_length)
        logger.info(f'{self.name.upper()}: Paillier key generated. ')

        self.dataset = None
        self.validset = None

        self.model = Model()

        self.passive_port = {}          # 被动方的名称 - 端口号对应
        self.revese_passive_port = {}   # 端口号 - 名称对应

        # 训练中临时变量
        self.cur_split_node = None      # 当前正在分裂的节点
        self.split_nodes = deque()      # 待分裂的节点队列
        self.cur_preds = None
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

        self.cur_preds = pd.Series(Calculator.init_pred, index=self.dataset.index)
        self.feature_split_time = pd.Series(0, index=self.dataset.index)
    
    def broadcast_pub_key(self):
        """
        将公钥广播给所有被动方
        """
        pub_dict = serialize_pub_key(self.pub_key)
        file_name = os.path.join(temp_root['file'][self.id], f'active_pub_key.json')
        with open(file_name, 'w+') as f:
            json.dump(pub_dict, f)

        @broadcast
        def send_pub_key(port: int, file_name: str):
            data = msg_name_file(self.name, file_name)
            logger.info(f'Sending public key. ')
            recv_data = requests.post(f'http://127.0.0.1:{port}/recvActivePubKey', data=data).json()
            self.passive_port[recv_data['party_name']] = port
            self.revese_passive_port[port] = recv_data['party_name']

        send_pub_key(file_name)
        logger.info(f'{self.name.upper()}: Public key broadcasted to all passive parties. ')

    def sample_align(self):
        """
        样本对齐
        """
        lock = threading.Lock()

        # 通知被动方计算样本列表的哈希值
        @broadcast
        def init_sample_align(port: int):
            logger.debug(f'{self.name.upper()}: Initiating sample alignment. ')
            requests.post(f'http://127.0.0.1:{port}/initSampleAlign')
        init_sample_align()
        
        # 主动方计算样本列表哈希值
        from utils.sha1 import sha1

        train_idx = self.dataset.index.tolist()
        train_idx_map = {sha1(idx): idx for idx in train_idx}
        train_hash = set(train_idx_map.keys())

        if self.validset is not None:
            valid_idx = self.validset.index.tolist()
            valid_idx_map = {sha1(idx): idx for idx in valid_idx}
            valid_hash = set(valid_idx_map.keys())
        
        # 轮询被动方，返回哈希后的样本列表，随后依次求交
        @poll
        def recv_sample_list(port: int) -> bool:
            nonlocal train_hash, valid_hash

            recv_data = requests.post(f'http://127.0.0.1:{port}/getSampleAlign').json()
            if '' in recv_data:       # 返回了空字典
                return False
            
            with open(recv_data['file_name'], 'r') as f:
                hash_data = json.load(f)
            logger.info(f'{self.name.upper()}: Received sample list from {recv_data["party_name"]}')
            
            lock.acquire()          # 加锁防止完整性破坏
            train_hash = train_hash.intersection(set(hash_data['train_hash']))
            if self.validset is not None:
                valid_hash = valid_hash.intersection(set(hash_data['valid_hash']))
            lock.release()
            return True
        recv_sample_list()
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
        @broadcast
        def send_aligned_sample(port: int, file_name: str):
            data = msg_name_file(self.name, file_name)
            logger.debug(f'Sending aligned sample: {data}.')
            requests.post(f'http://127.0.0.1:{port}/recvSampleList', data=data)
        send_aligned_sample(file_name)
        logger.info(f'{self.name.upper()}: Aligned sample broadcasted to all passive parties. ')

        self.dataset = self.dataset.loc[[train_idx_map[th] for th in train_hash], :]
        if self.validset is not None:
            self.validset = self.validset.loc[[valid_idx_map[vh] for vh in valid_hash], :]

    def train_status(self):
        """
        根据训练状态进行相关初始操作
        """
        if len(self.split_nodes) != 0:          # 还有待分裂节点，继续训练
            return True
        # 待分裂节点为空，则根据回归树的数量判断是否结束训练
        if len(self.model) < Calculator.max_trees:          # 树的数量未达到上限，则根据上一棵树更新数据，并将新一棵树加入
            logger.info(f'{self.name.upper()}: No pending node, creating new tree, index {len(self.model)}. ')
            new_root = self.create_new_tree()           # 更新权重并新建一棵树
            self.split_nodes.append(new_root)           # 加入待分裂节点队列
            self.model.append(new_root)                 # 加入模型
            self.update_gradients()         # 根据当前节点的预测值更新模型梯度
            return True
        else:
            logger.info(f'{self.name.upper()}: Training completed. ')
            self.create_new_tree()                  # 更新最后一棵树的叶子节点权重
            return False

    def create_new_tree(self):
        """
        向模型中新建一棵树，并计算上一棵树的叶子节点权重，更新对样本的预测值
        """
        if len(self.model) > 0:
            root = self.model[-1]
            for leaf in root.get_leaves():
                instance_space = leaf.instance_space
                leaf.weight = Calculator.leaf_weight(self.model.grad, self.model.hess, instance_space)        # 计算叶子节点的权重
                self.cur_preds[instance_space] += leaf.weight                               # 更新该叶子节点中所有样本的预测值
        else:
            self.cur_preds = pd.Series(Calculator.init_pred, index=self.dataset.index)

        new_root = TreeNode(0, self.dataset.index.tolist())
        return new_root

    def update_gradients(self):
        g, h = Calculator.grad(self.cur_preds, self.dataset)
        self.model.update_gradients(g, h, self.pub_key)

    def splits_score(self, instance_space: list) -> tuple:
        """
        主动方计算最佳分裂点，返回最佳分裂点信息
        """
        local_best_split = None
        for feature in [col for col in self.dataset.columns if col != 'y']:
            feature_values = self.dataset.loc[instance_space, feature].sort_values(ascending=True)
            split_indices = [int(qt * len(feature_values)) for qt in Calculator.quantile]

            for si in split_indices:
                left_space, right_space = feature_values.iloc[si:].index.tolist(), feature_values.iloc[:si].index.tolist()
                thresh = feature_values.iloc[si]
                split_score = Calculator.split_score_active(self.model.grad, self.model.hess, left_space, right_space)
                if not local_best_split or local_best_split[2] < split_score:
                    local_best_split = (feature, thresh, split_score, left_space)

        return local_best_split

    def passive_best_split_score(self, splits_file: str, full_grad_sum: float, full_hess_sum: float) -> tuple:
        """
        一个被动方的最佳分裂点增益
        """
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        
        local_best_split = None
        for idx, split in enumerate(splits_data):
            left_grad_sum, left_hess_sum = load_encrypted_number(split['grad_left'], self.pub_key), load_encrypted_number(split['hess_left'], self.pub_key)     # 反序列化
            left_grad_sum, left_hess_sum = self.pri_key.decrypt(left_grad_sum), self.pri_key.decrypt(left_hess_sum)         # 解密
            split_score = Calculator.split_score_passive(left_grad_sum, left_hess_sum, full_grad_sum, full_hess_sum)
            if not local_best_split or split_score > local_best_split[1]:
                local_best_split = (idx, split_score)
        
        return local_best_split

    def train(self):
        self.broadcast_pub_key()
        self.sample_align()
        while self.train_status():
            spliting_node = self.split_nodes.popleft()
            logger.info(f'{self.name.upper()}: Splitting node {spliting_node.id}. ')

            # 准备好本节点训练所用的文件
            instance_space_file = os.path.join(temp_root['file'][self.id], f'instance_space.json')
            instance_space = spliting_node.instance_space
            with open(instance_space_file, 'w+') as f:
                json.dump(instance_space, f)

            # 广播梯度
            @broadcast
            def send_gradients(port: int, instance_space_file: str):
                data = msg_gradient_file(self.name, instance_space_file, self.model.grad_file, self.model.hess_file)
                logger.info(f'Sending gradients. ')
                requests.post(f'http://127.0.0.1:{port}/recvGradients', data=data)
            send_gradients(instance_space_file)
            logger.info(f'{self.name.upper()}: Gradients broadcasted to all passive parties. ')

            local_best_split = self.splits_score(instance_space)
            global_splits = (self.name, 0, local_best_split[2])         # 全局最优分裂点 (训练方名称, 分裂点存储下标, 分裂增益)
            logger.info(f'{self.name.upper()}: Active best split point confirmed. ')

            full_grad_sum, full_hess_sum = self.model.grad[instance_space].sum(), self.model.hess[instance_space].sum()

            # 收集被动方的梯度信息
            @poll
            def get_splits_sum(port: int) -> bool:
                nonlocal global_splits
                recv_data = requests.post(f'http://127.0.0.1:{port}/getSplitsSum').json()
                if '' in recv_data:
                    return False
                logger.info(f'{self.name.upper()}: Received split sum from {self.revese_passive_port[port]}. ')
                passive_best_split = self.passive_best_split_score(recv_data['file_name'], full_grad_sum, full_hess_sum)
                if passive_best_split[1] > global_splits[2]:
                    global_splits = (self.revese_passive_port[port], passive_best_split[0], passive_best_split[1])
                return True
            get_splits_sum()
            logger.info(f'{self.name.upper()}: Global best split point confirmed, belongs to {global_splits[0]}. ')


class Model:
    def __init__(self, active_idx=0) -> None:
        self.trees = []

        # 原始 & 加密梯度，类型均为 pd.Series
        self.grad = None
        self.hess = None
        self.grad_enc = None
        self.hess_enc = None
        self.grad_file = os.path.join(temp_root['file'][active_idx], f'grad.pkl')
        self.hess_file = os.path.join(temp_root['file'][active_idx], f'hess.pkl')

    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        return self.trees[idx]
    
    def append(self, root):
        self.trees.append(root)
    
    def update_gradients(self, g, h, pub_key):
        """
        更新梯度并加密
        """
        self.grad = g
        self.hess = h
        self.encrypt_gradients(pub_key)
    
    def encrypt_gradients(self, pub_key: PaillierPublicKey):
        """
        将梯度用公钥加密
        """
        from tqdm import tqdm

        with tqdm(total=len(self.grad)*2) as pbar:

            def encrypt_data(data, pub_key: PaillierPublicKey):
                """
                将 data 加密后转换成字典形式返回
                """
                pbar.update(1)
                enc_data = pub_key.encrypt(data)
                return serialize_encrypted_number(enc_data)
            
            logger.info(f'Gradients encrypting... ')

            self.grad_enc, self.hess_enc = self.grad.apply(encrypt_data, pub_key=pub_key), self.hess.apply(encrypt_data, pub_key=pub_key)
            self.grad_enc.to_pickle(self.grad_file)
            self.hess_enc.to_pickle(self.hess_file)


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

    def get_leaves(self):
        if not self.left and not self.right:
            yield self
        if self.left:
            self.left.get_leaves()
        if self.right:
            self.right.get_leaves()