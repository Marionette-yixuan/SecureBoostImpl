import phe.util
import datetime

from phe import PaillierPublicKey, EncryptedNumber

from utils.log import logger


def serialize_encrypted_number(enc: EncryptedNumber) -> dict:
    """
    将加密数字对象序列化，用于存储在文件中
    """
    if enc.exponent > -32:
        enc = enc.decrease_exponent_to(-32)
        assert enc.exponent == -32
    
    return {
        'v': str(enc.ciphertext()), 
        'e': enc.exponent
    }

def load_encrypted_number(cipher_data: dict, pub_key: PaillierPublicKey) -> EncryptedNumber:
    """
    根据主动方公钥从字典中读取加密数字对象。
    """
    err_msg = 'Invalid cipher data. '
    assert 'v' in cipher_data, logger.error(err_msg)
    assert 'e' in cipher_data, logger.error(err_msg)

    enc = EncryptedNumber(
        public_key=pub_key, 
        ciphertext=int(cipher_data['v']), 
        exponent=cipher_data['e']
    )

    return enc

def serialize_pub_key(pub_key: PaillierPublicKey) -> dict:
    """
    将 PaillierPublicKey 转换为字典格式，用于存储。
    """
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    jwk_public = {
        'key': 'DAJ', 
        'alg': 'PAI-GN1', 
        'key_ops': ['encrypt'], 
        'n': phe.util.int_to_base64(pub_key.n), 
        'kid': f'Paillier public key generated by pheutil on {date}'
    }

    return jwk_public

def load_pub_key(pub_dict: dict) -> PaillierPublicKey:
    """
    从字典中读取 PaillierPublicKey
    """
    err_msg = 'Invalid public key. '
    assert 'alg' in pub_dict, logger.error(err_msg)
    assert pub_dict['alg'] == 'PAI-GN1', logger.error(err_msg)
    assert pub_dict['key'] == 'DAJ', logger.error(err_msg)

    n = phe.util.base64_to_int(pub_dict['n'])
    pub = phe.PaillierPublicKey(n)

    return pub
    