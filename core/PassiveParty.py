import os
import json
import pandas as pd
from utils.params import temp_root
from utils.encryption import load_pubkey, serialize_encrypted_number, load_encrypted_number
from msgs.messages import msg_name_file


class PassiveParty:
    def __init__(self, name: str) -> None:
        self.name = name                    # 被动方名称
        self.active_pubkey = None           # 主动方的公钥，用于数据加密
        self.dataset = None                 # 训练集
        self.validset = None                # 验证集

        self.look_up_table = pd.DataFrame(columns=['feature_name', 'feature_thresh'])           # 查找表

        # 训练中临时变量
        self.cur_splits = []                # 当前全部可能的分裂点信息
        self.feature_split_time = None      # 各特征的分裂次数

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

    def recv_active_pub_key(self, file_name: str):
        """
        保存主动方的公钥，用于梯度计算时的数据处理
        """
        with open(file_name, 'r+') as f:
            pub_dict = json.load(f)
        self.active_pubkey = load_pubkey(pub_dict)
        
    def get_sample_list(self) -> str:
        """
        向主动方返回加密后的样本列表文件
        """
        from utils.sha1 import sha1

        train_idx = self.dataset.index.tolist()
        train_idx_map = {sha1(idx): idx for idx in train_idx}
        train_hash = train_idx_map.keys()

        if self.validset is not None:
            valid_idx = self.validset.index.tolist()
            valid_idx_map = {sha1(idx): idx for idx in valid_idx}
            valid_hash = valid_idx_map.keys()

        map_data = {
            'train_map': train_idx_map, 
            'valid_map': valid_idx_map
        }
        map_file = os.path.join(temp_root['passive'], f'{self.name}_idx_map.json')
        with open(map_file, 'w+') as f:
            json.dump(map_data, f)

        json_data = {
            'train_hash': train_hash, 
            'valid_hash': valid_hash
        }

        file_name = os.path.join(temp_root['passive'], f'{self.name}_sample_align.json')
        with open(file_name, 'w+') as f:
            json.dump(json_data, f)

        return msg_name_file(self.name, file_name)
    
    def recv_sample_list(self, file_name: str) -> str:
        """
        根据主动方对齐后返回的样本列表文件更新本地数据集
        """
        with open(os.path.join(temp_root['passive'], f'{self.name}_idx_map.json'), 'r') as f:
            map_data = json.load(f)
        
        with open(file_name, 'r') as f:
            hash_data = json.load(f)

        train_idx = [map_data['train_map'][th] for th in hash_data['train_hash']]
        self.dataset = self.dataset.loc[train_idx, :]

        if self.validset is not None:
            valid_idx = [map_data['valid_map'][vh] for vh in hash_data['valid_hash']]
            self.validset = self.validset.loc[valid_idx, :]

