# File to convert the data-set into YOLO compatible format,
# as mentioned here - https://docs.ultralytics.com/datasets/detect/

from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import os
import json


def create_image_sets(level: str = 'extreme'):
    """
    Generate a set of image_ids that belong to train, val and test sets.
    :param level: The difficulty level to create the dataset on.
    :return: train_set, val_set and test_set - Sets containing image_ids
    """

    split_dir = os.path.join('splits', 'difficulty_splits', 'custom', level)
    split_names = ['train', 'valid', 'test']  # `val` for dataset, `valid` for this project
    train_set = set()
    val_set = set()
    test_set = set()

    for sp in split_names:
        with open(os.path.join(split_dir, sp + '.json'), 'r') as f:
            split_data = json.load(f)

        images = list(split_data.keys())
        for image in images:
            image_id = image.split('.')[0].split('/')[-1]
            if sp == 'train':
                train_set.add(image_id)
            elif sp == 'valid':
                val_set.add(image_id)
            else:
                test_set.add(image_id)

    return train_set, val_set, test_set


def create_dataset(level: str = 'extreme'):
    """
    Create the data/huggingface folder containing train, val and test sets
    :param level: The difficulty level to create the dataset on.
    :return: None
    """

    output_folder = os.path.join('data', 'huggingface')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create sub folders for splits
    splits = ['train', 'val', 'test']

    for s in splits:
        image_path = os.path.join(output_folder, s, 'images')
        label_path = os.path.join(output_folder, s, 'labels')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

    # Load the dataset
    dataset = load_dataset("Koshti10/Kitti-Images", split='train')  # By default, the images were loaded in train

    # Load the image id sets
    train_set, val_set, test_set = create_image_sets(level)

    # Iterate through the dataset and save images
    for idx, example in enumerate(dataset):
        # Get the image id
        img_id = example['id']

        img = example['image']
        label = example['text']

        if img_id in train_set:
            split_name = "train"
        elif img_id in val_set:
            split_name = "val"
        else:
            split_name = "test"

        # Save the data
        image_path = os.path.join(output_folder, split_name, 'images',  f'{img_id}.png')
        img.save(image_path)
        label_path = os.path.join(output_folder, split_name, 'labels', f'{img_id}.txt')
        label.save(label_path)

        if idx > 20:
            break


if __name__ == '__main__':
    create_dataset()
