import os
import json
import threading
import requests
import pandas as pd
from collections import deque
from phe import paillier, PaillierPublicKey

from core.Calculator import Calculator
from utils.log import logger
from utils.params import temp_root
from utils.encryption import serialize_pub_key, serialize_encrypted_number, load_encrypted_number
from utils.decorators import broadcast, use_thread, poll
from msgs.messages import msg_empty, msg_name_file, msg_gradient_file, msg_split_confirm


class ActiveParty:
    def __init__(self) -> None:
        Calculator.load_config()

        self.id = 0
        self.name = f'party{self.id}'
        self.pub_key, self.pri_key = paillier.generate_paillier_keypair(n_length=Calculator.keypair_gen_length)
        logger.info(f'{self.name.upper()}: Paillier key generated. ')

        self.dataset = None
        self.testset = None

        self.model = Model()

        self.passive_port = {}          # 被动方的名称 - 端口号对应
        self.reverse_passive_port = {}   # 端口号 - 名称对应

        # 训练中临时变量
        self.cur_split_node = None      # 当前正在分裂的节点
        self.split_nodes = deque()      # 待分裂的节点队列
        self.cur_preds = None
        self.feature_split_time = None  # 各特征的分裂次数

    def load_dataset(self, data_path: str, test_path: str=None):
        """
        加载数据集
        """
        dataset = pd.read_csv(data_path)
        dataset['id'] = dataset['id'].astype(str)
        self.dataset = dataset.set_index('id')

        if test_path:
            testset = pd.read_csv(test_path)
            testset['id'] = testset['id'].astype(str)
            self.testset = testset.set_index('id')

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
            self.reverse_passive_port[port] = recv_data['party_name']

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

        if self.testset is not None:
            valid_idx = self.testset.index.tolist()
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
            if self.testset is not None:
                valid_hash = valid_hash.intersection(set(hash_data.get('valid_hash', [])))
            lock.release()
            return True
        recv_sample_list()
        logger.info(f'{self.name.upper()}: Sample alignment finished. Intersect trainset contains {len(train_hash)} samples. ')

        json_data = {'train_hash': list(train_hash)}
        if self.testset is not None:
            json_data['valid_hash'] = list(valid_hash)

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
        if self.testset is not None:
            self.testset = self.testset.loc[[valid_idx_map[vh] for vh in valid_hash], :]

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
            logger.info(f'{self.name.upper()}: Accuracy after tree {len(self.model)-1}: {Calculator.accuracy(self.dataset["y"], self.cur_preds)}')          # 计算准确率
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
            # 检查叶子节点能否继续分裂（到达深度 / 样本过少都会停止分裂）
            if not spliting_node.splitable():
                continue
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

            # 主动方计算最佳分裂点
            local_best_split = self.splits_score(instance_space)
            global_splits = (self.name, 0, local_best_split[2])         # 全局最优分裂点 (训练方名称, 分裂点存储下标, 分裂增益)
            logger.info(f'{self.name.upper()}: Active best split point confirmed. ')

            full_grad_sum, full_hess_sum = self.model.grad[instance_space].sum(), self.model.hess[instance_space].sum()

            # 收集被动方的梯度信息，并确定最佳分裂点
            @poll
            def get_splits_sum(port: int) -> bool:
                nonlocal global_splits
                recv_data = requests.post(f'http://127.0.0.1:{port}/getSplitsSum').json()
                if '' in recv_data:
                    return False
                logger.info(f'{self.name.upper()}: Received split sum from {self.reverse_passive_port[port]}. ')
                passive_best_split = self.passive_best_split_score(recv_data['file_name'], full_grad_sum, full_hess_sum)
                if passive_best_split[1] > global_splits[2]:
                    global_splits = (self.reverse_passive_port[port], passive_best_split[0], passive_best_split[1])
                return True
            get_splits_sum()
            logger.info(f'{self.name.upper()}: Global best split point confirmed, belongs to {global_splits[0]} with gain {global_splits[2]}. ')
            if global_splits[2] < 0:                    # 如果分裂增益 < 0 则直接停止分裂
                continue
            

            # 根据最佳分裂点进行分裂 / 请求被动方确认
            if global_splits[0] == self.name:           # 最佳分裂点属于主动方
                feature_split = {
                    'feature_name': local_best_split[0], 
                    'feature_thresh': local_best_split[1]
                }
                left_space = local_best_split[3]
                param = {
                    'party_name': self.name, 
                    'left_space': left_space, 
                    'feature_split': feature_split
                }
                logger.info(f'{self.name.upper()}: Splitting on {feature_split["feature_name"]}. ')
            else:
                port = self.passive_port[global_splits[0]]      # 找到对应被动方的端口
                data = msg_split_confirm(global_splits[0], global_splits[1])
                recv_data = requests.post(f'http://127.0.0.1:{port}/confirmSplit', data=data).json()
                look_up_id = recv_data['split_index']
                logger.debug(f'Received confirm data, look up index: {look_up_id}. ')

                # 获得分裂后的左空间
                with open(recv_data['left_space'], 'r') as f:
                    left_space = json.load(f)
                param = {
                    'party_name': global_splits[0], 
                    'left_space': left_space, 
                    'look_up_id': look_up_id
                }
            left_node, right_node = spliting_node.split(**param)
            self.split_nodes.extend([left_node, right_node])

    def dump_model(self, file_path: str):
        """
        将模型存储到指定路径
        """
        # 判断给定路径是文件夹还是文件
        dir_path, file_name = os.path.split(file_path)
        if not os.path.exists(dir_path):
            logger.error(f'{self.name.upper()}: Model saving path not exists: \'{dir_path}\'. ')
            return
        if not file_name:
            import time
            file_path = os.path.join(file_path, time.strftime('model%m%d%H%M.json', time.localtime()))

        with open(file_path, 'w+') as f:
            logger.debug(self.model.dump())
            json.dump(self.model.dump(), f)
        logger.info(f'{self.name.upper()}: Model dumped to {file_path}. ')
        return file_path

    def load_model(self, file_path: str):
        """
        从指定路径加载模型
        """
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        for tree_data in model_data:
            self.model.load(tree_data)
        logger.info(f'{self.name.upper()}: Model loaded from {file_path}. ')

    def predict(self):
        """
        对测试集进行预测
        """
        if self.testset is None:
            logger.error(f'{self.name.upper()}: No testset loaded. ')
            return
        
        logger.info(f'{self.name.upper()}: Start predicting. ')
        self.split_nodes = deque()
        preds = pd.Series(Calculator.init_pred, index=self.testset.index)

        for tree in self.model:
            tree.instance_space = self.testset.index.tolist()
            self.split_nodes.append(tree)               # 树根入队
            while len(self.split_nodes):
                spliting_node = self.split_nodes.popleft()
                instance_space = spliting_node.instance_space
                if not spliting_node.left:                              # 为叶子节点
                    preds[instance_space] += spliting_node.weight
                    continue
                elif spliting_node.party_name == self.name:             # 为主动方的分裂节点
                    # 分裂样本空间
                    logger.info(f'{self.name.upper()}: Splitting on node {spliting_node.id} from {spliting_node.party_name}. ')
                    feature_name, feature_thresh = spliting_node.feature_split['feature_name'], spliting_node.feature_split['feature_thresh']
                    left_space = self.testset.loc[instance_space, feature_name] >= feature_thresh
                    left_space = left_space[left_space].index.tolist()
                    right_space = [idx for idx in instance_space if idx not in left_space]
                    
                    spliting_node.left.instance_space = left_space
                    spliting_node.right.instance_space = right_space

                    self.split_nodes.extend([spliting_node.left, spliting_node.right])
                else:                                                   # 为被动方的分裂节点
                    logger.info(f'{self.name.upper()}: Splitting on node {spliting_node.id} from {spliting_node.party_name}. ')
                    party_name = spliting_node.party_name
                    look_up_id = spliting_node.look_up_id
                    instance_space_file = os.path.join(temp_root['file'][self.id], f'instance_space.json')
                    with open(instance_space_file, 'w+') as f:
                        json.dump(instance_space, f)

                    @use_thread
                    def get_passive_split(party_name: str, instance_space_file: str, look_up_id: int):
                        port = self.passive_port[party_name]
                        data = msg_name_file(self.name, instance_space_file, look_up_id)
                        logger.info(f'{self.name.upper()}: Sending prediction request to {party_name}. ')
                        recv_data = requests.post(f'http://127.0.0.1:{port}/getPassiveSplit', data=data).json()

                        with open(recv_data['file_name'], 'r') as f:
                            split_space = json.load(f)
                        left_space, right_space = split_space['left_space'], split_space['right_space']

                        spliting_node.left.instance_space = left_space
                        spliting_node.right.instance_space = right_space

                        self.split_nodes.extend([spliting_node.left, spliting_node.right])
                    
                    get_passive_split(party_name, instance_space_file, look_up_id)

        logger.info(f'{self.name.upper()}: Test accuracy: {Calculator.accuracy(self.testset["y"], preds)}. ')
        logger.info(f'{self.name.upper()}: All finished. ')

 
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

    def dump(self) -> str:
        data = [tree.dump() for tree in self.trees]
        return data

    def load(self, tree_dict: dict):
        """
        将字典中的数据加载成一棵新树加入模型
        """
        root = TreeNode().load(tree_dict)
        self.trees.append(root)


class TreeNode:
    def __init__(self, id: int=0, instance_space: list=None) -> None:
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

    def dump(self):
        """
        将以 self 为树根的树存储在字典中
        """
        data = { 'id': self.id }
        if self.left:               # 左右子树一定同时存在或同时不存在，判断左子树即可。不为叶子节点，说明有分裂信息，没有权重
            data['party_name'] = self.party_name

            if self.feature_split:
                data['feature_split'] = self.feature_split
            else:
                data['look_up_id'] = self.look_up_id
            
            data['left'] = self.left.dump()
            data['right'] = self.right.dump()
        else:
            data['weight'] = self.weight

        return data

    def load(self, tree_dict: dict):
        """
        根据字典中的数据加载树
        """
        self.id = tree_dict['id']
        if 'party_name' in tree_dict:            # 不为叶子节点
            self.party_name = tree_dict['party_name']
            if 'feature_split' in tree_dict:
                self.feature_split = tree_dict['feature_split']
            else:
                self.look_up_id = tree_dict['look_up_id']
            self.left = TreeNode(0, [])
            self.left.load(tree_dict['left'])
            self.right = TreeNode(0, [])
            self.right.load(tree_dict['right'])
        else:
            self.weight = tree_dict['weight']

        return self

    def splitable(self):
        """
        当节点的标号达到了足够的深度，或者节点的样本数量足够少时，不再尝试分裂
        """
        if self.id >= 2 ** (Calculator.max_depth - 1) - 1:
            return False
        if len(self.instance_space) < Calculator.min_sample:
            return False
        return True

    def split(self, party_name, left_space, feature_split=None, look_up_id=0):
        right_space = list(set(self.instance_space) - set(left_space))
        logger.debug(f'Left space: {len(left_space)}, right space: {len(right_space)}')
        self.party_name = party_name
        self.feature_split = feature_split
        self.look_up_id = look_up_id
        self.left, self.right = TreeNode(self.id * 2 + 1, left_space), TreeNode(self.id * 2 + 2, right_space)
        return self.left, self.right

    def get_leaves(self):
        if not self.left and not self.right:
            yield self
        if self.left:
            yield from self.left.get_leaves()
        if self.right:
            yield from self.right.get_leaves()

        