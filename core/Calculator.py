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