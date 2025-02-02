import pandas as pd
import os
import matplotlib.pyplot as plt

SAVE_DIR = 'splits'
SUB_DIR = 'csvs'
SAVE_NAME = 'labels.csv'

if not os.path.exists(os.path.join(SAVE_DIR, SUB_DIR)):
    os.makedirs(os.path.join(SAVE_DIR, SUB_DIR))


def create_df():
    """
    :param: None    :return:
    Save a Dataframe that has all the labels form the Kitti Dataset.
    """

    # Get paths of text files that contains label for each image
    LABEL_DIR = os.path.join('data', 'labels')
    label_txt_files = os.listdir(LABEL_DIR)

    # Collect all labels in a dataframe
    # More information about the labels can be found in the `devkit_object` available here - https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
    # Initialize an empty DataFrame with column types
    columns = ['label', 'truncated', 'occluded', 'alpha', 'left-top-x', 'left-top-y',
               'right-bottom-x', 'right-bottom-y', 'height', 'width', 'length',
               'x', 'y', 'z', 'rotation_y', 'image']
    labels_df = pd.DataFrame(columns=columns)

    # Collect each row entry
    rows = []
    for txt_file in label_txt_files:
        txt_file_path = os.path.join(LABEL_DIR, txt_file)
        image_number = txt_file.split(".")[0]
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()
            row_dict = {'label': values[0], 'truncated': float(values[1]), 'occluded': int(values[2]),
                        'alpha': float(values[3]), 'left-top-x': float(values[4]), 'left-top-y': float(values[5]),
                        'right-bottom-x': float(values[6]), 'right-bottom-y': float(values[7]),
                        'height': float(values[8]), 'width': float(values[9]), 'length': float(values[10]),
                        'x': float(values[11]), 'y': float(values[12]), 'z': float(values[13]),
                        'rotation_y': float(values[14]), 'image': str(image_number)}
            rows.append(row_dict)

    data_df = pd.DataFrame(rows, columns=columns)
    labels_df = pd.concat([labels_df, data_df], ignore_index=True)
    labels_df.to_csv(os.path.join(SAVE_DIR, SUB_DIR, SAVE_NAME), index=False)
    print("Labels collected from text files and saved to {}".format(os.path.join(SAVE_NAME)))

    return None


def plot_label_distr(df: pd.DataFrame):
    """
    :param df: Dataframe that has all the labels form the Kitti Dataset.
    :return: None
    Plot the distribution of the labels by count
    """

    # Calculate counts and percentages
    label_counts = df['label'].value_counts()
    label_percentages = label_counts / label_counts.sum() * 100

    # Plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(label_counts.index, label_counts.values, color='skyblue')

    # Annotate percentages and counts on top of bars
    for bar, count, percentage in zip(bars, label_counts.values, label_percentages):
        height = bar.get_height()
        label_text = f'{percentage:.2f}% ({count})'
        plt.text(bar.get_x() + bar.get_width() / 2, height, label_text,
                 ha='center', va='bottom')

    plt.title('Distribution of Labels with Percentages and Counts')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    return None


def categorize_difficulty(row):
    """
    :param row: Row entry from the dataframe containing labels
    :return: List of difficulty labels
    """
    difficulties = []
    if row['label'] == 'Car':  # Distribute only for car
        top_x = row['left-top-x']
        bottom_x = row['right-bottom-x']
        height = abs(top_x - bottom_x)
        occlusion = row['occluded']
        truncation = row['truncated']

        if height >= 40 and occlusion == 0 and truncation <= 0.15:
            difficulties.append('Easy')
        if height >= 25 and occlusion <= 1 and truncation <= 0.30:
            difficulties.append('Moderate')
        if height >= 25 and occlusion <= 2 and truncation <= 0.50:
            difficulties.append('Hard')

        difficulties.append('Extreme')  # Add a new difficulty level without any thresholds
    else:
        difficulties.append('None')

    return difficulties

def plot_by_difficulty(df: pd.DataFrame):
    """
    :param df: Dataframe containing collected labels
    :return: None
    Plot the distribution of the labels by difficulty
    """

    df['difficulty'] = df.apply(categorize_difficulty, axis=1)
    difficulties = [difficulty for sublist in df['difficulty'] for difficulty in sublist]

    # Create a DataFrame for plotting
    difficulty_df = pd.DataFrame(difficulties, columns=['difficulty'])
    label_counts = difficulty_df['difficulty'].value_counts()
    label_percentages = label_counts / label_counts.sum() * 100

    plt.figure(figsize=(12, 7))
    bars = plt.bar(label_counts.index, label_counts.values, color='skyblue')

    # Annotate percentages and counts
    for bar, count, percentage in zip(bars, label_counts.values, label_percentages):
        height = bar.get_height()
        label_text = f'{percentage:.2f}% ({count})'
        plt.text(bar.get_x() + bar.get_width() / 2, height, label_text,
                 ha='center', va='bottom')

    plt.title('Distribution of Difficulty for Cars')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Count')
    plt.show()

    return None


if __name__ == '__main__':

    #  A dataframe as a CSV is already provided here
    create_df()

    df = pd.read_csv(os.path.join(SAVE_DIR, SUB_DIR, SAVE_NAME))

    # Plot label counts
    plot_label_distr(df)

    # Plot by difficulty level based on occlusion
    plot_by_difficulty(df)
