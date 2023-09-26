import pandas as pd
import numpy as np
import configparser

from utils.log import logger


class Calculator:
    ## 训练时使用的参数，设计为该类的静态变量
    # 计算梯度
    lmd = None
    gma = None
    # 样本空间的划分分位数
    quantile = None
    # 节点的最小样本数量（如果低于该数量则不会再分裂）
    min_sample = None
    # 模型参数
    max_depth = None       # 树最大深度
    max_trees = None       # 最大树数量
    # 预测参数
    init_pred = None             # 初始预测值
    output_thresh = None        # 对于输出的映射，大于该值则输出 1，反之输出 0
    # 其它参数
    keypair_gen_length = None     # Paillier 公私钥的生成长度

    def __init__(self) -> None:
        pass

    @staticmethod
    def grad(preds: pd.Series, trainset: pd.DataFrame) -> (pd.Series, pd.Series):
        """
        log 损失函数下的梯度
        """
        assert 'y' in trainset.columns, logger.error('Trainset doesn\'t contain columns \'y\'')
        labels = trainset['y'].values
        preds = 1.0 / (1.0 + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1 - preds)
        logger.debug(f'Gradients: {list(grad.iloc[:5].values)}. ')
        return grad, hess

    @staticmethod
    def split_score_active(grad: pd.Series, hess: pd.Series, left_space: list, right_space: list) -> float:
        """
        根据梯度和划分后的样本空间计算分裂增益。该方法用于主动方（被动方无法获得可以用来进行乘除运算的梯度）。
        """
        left_grad_sum = grad[left_space].sum()
        left_hess_sum = hess[left_space].sum()
        full_grad_sum = grad[left_space + right_space].sum()
        full_hess_sum = hess[left_space + right_space].sum()

        return Calculator.split_score_passive(left_grad_sum, left_hess_sum, full_grad_sum, full_hess_sum)
    
    @staticmethod
    def split_score_passive(left_grad_sum: float, left_hess_sum: float, full_grad_sum: float, full_hess_sum: float) -> float:
        """
        根据计算好的左样本空间和以及整个空间和计算分裂增益。该方法用于被动方传来的数据计算增益。
        """
        right_grad_sum = full_grad_sum - left_grad_sum
        right_hess_sum = full_hess_sum - left_hess_sum
        
        temp = (left_grad_sum ** 2) / (left_hess_sum + Calculator.lmd)
        temp += (right_grad_sum ** 2) / (right_hess_sum + Calculator.lmd)
        temp -= (full_grad_sum ** 2) / (full_hess_sum + Calculator.lmd)
        return temp / 2 - Calculator.gma

    @staticmethod
    def leaf_weight(grad: pd.Series, hess: pd.Series, instance_space: list) -> float:
        """
        计算叶子节点的权重
        """
        return -grad[instance_space].sum() / (hess[instance_space].sum() + Calculator.lmd)
    
    @staticmethod
    def accuracy(y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        计算准确率指标
        """
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred = y_pred.apply(lambda x: 1 if x > Calculator.output_thresh else 0)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    
    @staticmethod
    def load_config(config_path='config/config.conf'):
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        Calculator.lmd = float(cfg['params.train']['lmd'])
        Calculator.gma = float(cfg['params.train']['gma'])
        Calculator.quantile = np.linspace(0, 1, int(cfg['params.train']['quantile']) + 1).tolist()[1:-1]
        Calculator.min_sample = int(cfg['params.train']['min_sample'])
        Calculator.max_depth = int(cfg['params.train']['max_depth'])
        Calculator.max_trees = int(cfg['params.train']['max_trees'])
        Calculator.init_pred = float(cfg['params.predict']['init_pred'])
        Calculator.output_thresh = float(cfg['params.predict']['output_thresh'])
        Calculator.keypair_gen_length = int(cfg['encryption']['keypair_gen_length'])
        logger.debug(f'Config loaded. Trees and depth: ({Calculator.max_trees}, {Calculator.max_depth}). ')

    @staticmethod
    def generate_config(config_path='config/config.conf'):
        cfg = configparser.ConfigParser()
        cfg['params.train'] = {
            'lmd': 10.0,
            'gma': 0.0, 
            'quantile': 8, 
            'min_sample': 20,
            'max_depth': 5,
            'max_trees': 4
        }
        cfg['params.predict'] = {
            'init_pred': 0.0,
            'output_thresh': 0.33
        }
        cfg['encryption'] = {
            'keypair_gen_length': 192
        }
        with open(config_path, 'w+') as config_file:
            cfg.write(config_file)


if __name__ == '__main__':
    Calculator.generate_config()        # 先注释上面的 import utils.logger 再运行文件
