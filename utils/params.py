import os

temp_root = {
    dir_type: [
            os.path.join('temp', dir_type, f'party-{idx}') for idx in range(10)
        ] for dir_type in ['file', 'model']
    }


if __name__ == '__main__':
    print(temp_root)