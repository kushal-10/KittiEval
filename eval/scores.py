# import os
# import ast
# import json
#
#
# RES_DIR = 'results'
# dirs = os.listdir(RES_DIR)
#
# # Collect result JSONs
# json_paths = []
# for dir in dirs:
#     jsons = os.listdir(os.path.join(RES_DIR, dir))
#     for json in jsons:
#         json_paths.append(os.path.join(RES_DIR, dir, json))
#
# def parse_bbox(bbox_str):
#     # Use ast.literal_eval to safely parse the string representation of the list of lists
#     try:
#         bbox_list = ast.literal_eval(bbox_str)
#         # Ensure the parsed result is a list of lists of floats
#         if isinstance(bbox_list, list) and all(isinstance(sublist, list) for sublist in bbox_list):
#             return [list(map(float, sublist)) for sublist in bbox_list]
#         else:
#             raise ValueError("Parsed data is not in the expected format.")
#     except (ValueError, SyntaxError) as e:
#         raise ValueError(f"Error parsing bounding box string: {e}")
#
#
# def get_score(json_path: str):
#     """
#     :param json_path: Path to json file
#     :return:
#     """
#
#     instance = json_path.split('/')[-1].split('.')[0]
#     with open(json_path, 'r') as file:
#         data = json.load(file)
#
#
#
#
#
#
#     return instance
#
#
# if __name__ == '__main__':
#     get_score(json_paths[0])