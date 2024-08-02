## Create splits by difficulty level - easy, moderate, hard
import pandas as pd
import os
import json
from tqdm import tqdm

BASE_DIR = 'splits'
SUB_DIR = 'csvs'
SPLIT_DIR = 'difficulty_splits'


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

    # Additional difficulty level, considering extreme cases
    difficulties.append('extreme')

    return difficulties


def check_instance_custom(df_row):
    """
    Check in which case to classify the current instance, based on a custom criteria
    This excludes occlusion and truncation from checks, and only considers min pixel height

    :param df_row: Row instance from the dataframe containing a particular split (train/test/valid)
    :return: difficulty classification
    """

    occlusion = df_row['occluded']
    truncation = df_row['truncated']
    top_y = df_row['left-top-y']
    bottom_y = df_row['right-bottom-y']
    height = abs(top_y - bottom_y)

    difficulties = []
    if height >= 40:
        difficulties.append('easy')

    if height >= 30:
        difficulties.append('moderate')

    if height >= 25:
        difficulties.append('hard')

    # Additional difficulty level, considering extreme cases
    difficulties.append('extreme')

    return difficulties


def save_difficulty_splits(difficulty_type: str = 'base'):
    """
    Save splits based on difficulty type passed, containing train/test/valid data
    :param difficulty_type: Type of difficulty classification [ 'base' or 'custom']
    :return: None
    """

    main_splits = ['train', 'valid', 'test']
    levels = ['easy', 'moderate', 'hard', 'extreme']
    valid_labels = ['Car', 'DontCare']

    for level in tqdm(levels, desc=f"Creating Difficulty level splits for difficulty type - {difficulty_type}, splits - Easy, Moderate, Hard and Extreme"):
        for split in main_splits:
            split_data = {}
            df = pd.read_csv(os.path.join(BASE_DIR, SUB_DIR, split + "_split.csv"))
            for i in range(len(df)):
                df_row = df.iloc[i]
                label = df_row['label']

                if difficulty_type == 'base':
                    difficulties = check_instance(df_row)
                else:
                    difficulties = check_instance_custom(df_row)

                if level in difficulties:
                    # Add this to the difficulty level data
                    image = str(df_row['image'])
                    while len(image) != 6:
                        image = "0" + image
                    image += ".png"
                    image_path = os.path.join('data', 'images', image)

                    observation = {label: [df_row['left-top-x'], df_row['left-top-y'], df_row['right-bottom-x'],
                                           df_row['right-bottom-y']]}
                    if image_path not in split_data:
                        split_data[image_path] = []

                    # Add observations only for valid_label [DontCare and Car]
                    if label in valid_labels:
                        split_data[image_path].append(observation)

            SAVE_DIR = os.path.join(BASE_DIR, SPLIT_DIR, difficulty_type, level)
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)

            save_path = os.path.join(SAVE_DIR, split + ".json")
            with open(save_path, 'w') as file:
                json.dump(split_data, file, indent=4)


if __name__ == '__main__':
    save_difficulty_splits('base')
    save_difficulty_splits('custom')
