import json


def msg_empty():
    """
    不需要包含任何信息的消息
    """
    return {'': ''}

def msg_name_file(party_name: str, file_name: str):
    """
    所有只需要传递训练方名称和文件名的消息
    """
    res_dict = {
        'party_name': party_name, 
        'file_name': file_name
    }
    return res_dict

def msg_gradient_file(party_name: str, instance_space_file: str, grad_file: str, hess_file: str):
    """
    传输节点的样本空间和导数所存储的文件名消息
    """
    res_dict = {
        'party_name': party_name, 
        'instance_space': instance_space_file, 
        'grad': grad_file, 
        'hess': hess_file
    }
    return res_dict