# File to convert the data-set into YOLO compatible format,
# as mentioned here - https://docs.ultralytics.com/datasets/detect/

from datasets import load_dataset
import os
import json
import shutil
from PIL import Image
from tqdm import tqdm


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


def convert_bbox_format(image_width, image_height, x1, y1, x2, y2):
    """
    Convert bounding box coordinates from (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
    to YOLO format (center_x, center_y, width, height).
    :param image_width: The width of the image
    :param image_height: The height of the image
    :param x1: The top left x coordinate of the bounding box
    :param y1: The top left y coordinate of the bounding box
    :param x2: The bottom right x coordinate of the bounding box
    :param y2: The bottom right y coordinate of the bounding box
    :return: A new bounding box (normalized) formatted as (center_x, center_y, width, height)
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    # Normalize coordinates
    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    return center_x, center_y, width, height


def process_label_file(input_path, output_path, image_width, image_height):
    """
    Process the label file according to the requirements and save the new formatted file.
    :param input_path: The path to the label file
    :param output_path: The path to the output file
    :param image_width: The width of the image
    :param image_height: The height of the image
    """
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if parts[0] == 'Car':
                # Convert 'Car' to '1'
                class_id = '1'
                # Extract columns 5, 6, 7, 8 and convert them to YOLO format
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                # Convert to YOLO format
                center_x, center_y, width, height = convert_bbox_format(image_width, image_height, left, top, right, bottom)
                # Write to the output file in the required format
                outfile.write(f"{class_id} {center_x} {center_y} {width} {height}\n")


def create_dataset(level: str = 'extreme'):
    """
    Create the data/huggingface folder containing train, val and test sets
    :param level: The difficulty level to create the dataset on.
    :return: None
    """

    output_folder = os.path.join('data', 'huggingface')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a backup folder that contains all original labels for the splits
    # Used in evaluation
    backup_folder = os.path.join(output_folder, 'backup')
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    # Create sub folders for splits
    splits = ['train', 'val', 'test']

    for s in splits:
        image_path = os.path.join(output_folder, s, 'images')
        label_path = os.path.join(output_folder, s, 'labels')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        backup_label_path = os.path.join(backup_folder, s, 'labels')
        if not os.path.exists(backup_label_path):
            os.makedirs(backup_label_path)

    # Load the dataset
    dataset = load_dataset("Koshti10/Kitti-Images", split='train')  # By default, the images were loaded in train

    # Load the image id sets
    train_set, val_set, test_set = create_image_sets(level)

    # Iterate through the dataset and save images
    for example in tqdm(dataset, desc='Creating YOLO dataset'):
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
        img_pth = os.path.join(output_folder, split_name, 'images',  f'{img_id}.png')
        img.save(img_pth)

        with Image.open(img_pth) as img:
            image_width, image_height = img.size

        lbl_pth = os.path.join(output_folder, split_name, 'labels', f'{img_id}.txt')
        process_label_file(label, lbl_pth, image_width, image_height)

        back_lbl_pth = os.path.join(backup_folder, split_name, 'labels', f'{img_id}.txt')
        shutil.copy(label, back_lbl_pth)


if __name__ == '__main__':
    create_dataset()
