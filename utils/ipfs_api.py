import requests
from utils.log import logger


def ipfs_upload(file_path):
    url = 'http://127.0.0.1:5001/api/v0/add'
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        cid = response.json()['Hash']
        logger.debug(f"Upload {file_path} to IPFS successfully. File CID: {cid}")
        return cid
    else:
        logger.error(f"Upload {file_path} to IPFS failed. Response: {response.json()}")
        return None
    

def ipfs_download(cid):
    url = f'http://127.0.0.1:5001/api/v0/cat?arg={cid}'
    response = requests.post(url)
    if response.status_code == 200:
        file_content = response.content
        return file_content
    else:
        logger.error(f"Download {cid} from IPFS failed. Response: {response.json()}")
        return None