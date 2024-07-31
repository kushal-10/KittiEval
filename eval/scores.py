import os
import ast
import json


RES_DIR = 'results'
dirs = os.listdir(RES_DIR)

# Collect result JSONs
json_paths = []
for d in dirs:
    jsons = os.listdir(os.path.join(RES_DIR, d))
    for json in jsons:
        json_paths.append(os.path.join(RES_DIR, d, json))


def get_score(json_path: str):
    """
    :param json_path: Path to json file
    :return:
    """

    instance = json_path.split('/')[-1].split('.')[0]
    with open(json_path, 'r') as file:
        split_data = json.load(file)

    return instance


if __name__ == '__main__':
    get_score(json_paths[0])