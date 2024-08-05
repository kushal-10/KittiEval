# Hotfix file to replace the label entry from 1 to 0 in labels/*.txt.
# Not required, fixed main code in `create_yolo_dataset.py`

import os
from tqdm import tqdm

def replace_first_entry_in_file(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the first entry in each line
    new_lines = []
    for line in lines:
        parts = line.split()
        if parts:
            parts[0] = '0'
            new_line = ' '.join(parts)
            new_lines.append(new_line + '\n')

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)


def process_directory(directory_path):
    # Iterate through each file in the directory
    for filename in tqdm(os.listdir(directory_path), desc='Processing directory - {}'.format(directory_path)):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            replace_first_entry_in_file(file_path)


splits = ['train', 'val', 'test']

for split in splits:
    dir_path = os.path.join('data', 'huggingface', split, 'labels')
    process_directory(dir_path)
