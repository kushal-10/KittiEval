from typing import Tuple
import pandas as pd
import os
from sklearn.model_selection import train_test_split

SAVE_DIR = 'splits'
SUB_DIR = 'csvs'

def splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param df: Dataframe formed by collecting labels from text files
    :return: train, validation and test dataframes
    """

    # Group by 'image' and aggregate data
    # This ensures no images are overlapping between the splits
    grouped = df.groupby('image')
    images = list(grouped.groups.keys())

    # Split the images into train, valid, and test sets
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)  # 70 - 30 split
    valid_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)  # 50-50 split (from rem 30%)

    # Create dataframes for train, valid, and test sets
    train_df = df[df['image'].isin(train_images)]
    valid_df = df[df['image'].isin(valid_images)]
    test_df = df[df['image'].isin(test_images)]

    return train_df, valid_df, test_df


def check_splits(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    """
    :param train_df: Train split dataframe
    :param valid_df: Validation split dataframe
    :param test_df: Test split dataframe
    :return: count of overlapping images, should be 0
    """

    # Extract unique image names from each DataFrame
    train_images = set(train_df['image'])
    valid_images = set(valid_df['image'])
    test_images = set(test_df['image'])

    # Check for overlaps
    train_valid_overlap = train_images.intersection(valid_images)
    train_test_overlap = train_images.intersection(test_images)
    valid_test_overlap = valid_images.intersection(test_images)

    # Print the results
    print(f"Number of overlapping images between train and valid sets: {len(train_valid_overlap)}")
    print(f"Number of overlapping images between train and test sets: {len(train_test_overlap)}")
    print(f"Number of overlapping images between valid and test sets: {len(valid_test_overlap)}")

    return len(train_test_overlap) + len(valid_test_overlap) + len(train_valid_overlap)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(SAVE_DIR, SUB_DIR, 'labels.csv'))
    train_df, valid_df, test_df = splits(df)

    # Final check that each split has unique image
    assert check_splits(train_df, valid_df, test_df)==False

    train_df.to_csv(os.path.join(SAVE_DIR, SUB_DIR, 'train_split.csv'), index=False)
    valid_df.to_csv(os.path.join(SAVE_DIR, SUB_DIR, 'valid_split.csv'), index=False)
    test_df.to_csv(os.path.join(SAVE_DIR, SUB_DIR, 'test_split.csv'), index=False)

    save_path = os.path.join(SAVE_DIR, SUB_DIR)
    print(f"Splits created and saved under {save_path}")
