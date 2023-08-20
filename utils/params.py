import os

temp_root = {file_type: os.path.join('temp', file_type) for file_type in ['active', 'passive', 'model']}