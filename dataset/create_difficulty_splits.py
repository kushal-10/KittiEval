## Create splits by difficulty level - easy, moderate, hard
import pandas as pd
import os
import json
from tqdm import tqdm

def check_instance(df_row):
    """
    Check in which case to classify the current instance
    :param df_row: Row instance from the dataframe containing a particular split (train/test/valid)
    :return: difficulty classification
    """

    occlusion = df_row['occluded']
    truncation = df_row['truncated']
    top_y = df_row['left-top-y']
    bottom_y = df_row['right-bottom-y']
    height = abs(top_y - bottom_y)

    difficulties = []
    if height >= 40 and occlusion <= 0 and truncation <= 0.15:
        difficulties.append('easy')

    if height >= 25 and occlusion <= 1 and truncation <= 0.3:
        difficulties.append('moderate')

    if height >= 25 and occlusion <= 2 and truncation <= 0.5:
        difficulties.append('hard')

    return difficulties


def save_difficulty_splits():
    """
    Save splits based on difficulties, containing train/test/valid data
    :return: None
    """

    main_splits = ['train', 'valid', 'test']
    levels = ['easy', 'moderate', 'hard']
    valid_labels = ['Car', 'DontCare']

    for level in tqdm(levels, desc="Creating Difficulty level splits for Easy, Moderate, and Hard"):
        for split in main_splits:
            split_data = {}
            df = pd.read_csv(os.path.join('dataset', 'csvs', split+"_split.csv"))
            for i in range(len(df)):
                df_row = df.iloc[i]
                label = df_row['label']

                difficulties = check_instance(df_row)
                if level in difficulties:
                    # Add this to the difficulty level data
                    image = str(df_row['image'])
                    while len(image) != 6:
                        image = "0"+image
                    image += ".png"
                    image_path = os.path.join('data', 'images', image)

                    observation = {label: [df_row['left-top-x'], df_row['left-top-y'], df_row['right-bottom-x'], df_row['right-bottom-y']]}
                    if image_path not in split_data:
                        split_data[image_path] = []

                    # Add observations only for valid_label [DontCare and Car]
                    if label in valid_labels:
                        split_data[image_path].append(observation)

            save_path = os.path.join('dataset', 'jsons', level+"_"+split+".json")
            with open(save_path, 'w') as file:
                json.dump(split_data, file, indent=4)


if __name__ == '__main__':
    RES_DIR = os.path.join('dataset', 'jsons')
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    save_difficulty_splits()
