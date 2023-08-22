import pandas as pd
import numpy as np

from utils.log import logger


class Calculator:
    ## 训练时使用的参数，设计为该类的静态变量
    # 计算梯度
    lmd = 30.0
    gma = 0.0
    # 样本空间的划分分位数
    quantile = None         # TODO
    # 节点的最小样本数量（如果低于该数量则不会再分裂）
    min_sample = 8
    # 模型参数
    max_depth = 5       # 树最大深度
    max_trees = 4       # 最大树数量
    # 预测参数
    init_pred = 0.0             # 初始预测值
    output_thresh = 0.33        # 对于输出的映射，大于该值则输出 1，反之输出 0
    # 其它参数
    keypair_gen_length = 256    # Paillier 公私钥的生成长度

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
