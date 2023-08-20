import json


def msg_name_file(party_name: str, file_name: str):
    """
    所有只需要传递训练方名称和文件名的消息
    """
    res_dict = {
        'party_name': party_name, 
        'file_name': file_name
    }
    return json.dumps(res_dict)