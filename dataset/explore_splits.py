import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

# Define file paths
file_paths = {
    'train_easy': 'dataset/csvs/train_easy_split.csv',
    'train_moderate': 'dataset/csvs/train_moderate_split.csv',
    'train_hard': 'dataset/csvs/train_hard_split.csv',
    'test_easy': 'dataset/csvs/test_easy_split.csv',
    'test_moderate': 'dataset/csvs/test_moderate_split.csv',
    'test_hard': 'dataset/csvs/test_hard_split.csv',
    'valid_easy': 'dataset/csvs/valid_easy_split.csv',
    'valid_moderate': 'dataset/csvs/valid_moderate_split.csv',
    'valid_hard': 'dataset/csvs/valid_hard_split.csv'
}

# Load DataFrames
dfs = {key: pd.read_csv(path) for key, path in file_paths.items()}

def parse_bbox(bbox_str):
    bbox_str = bbox_str.replace('np.float64(', '').replace(')', '')
    bbox_list = re.findall(r"([0-9.]+)", bbox_str)
    # Convert to floats
    return [float(x) for x in bbox_list]

def plot_distributions():

    # Parse bbox (convert from string)
    for df in dfs.values():
        df['bbox'] = df['bbox'].apply(parse_bbox)

    image_counts = {key: df['image'].nunique() for key, df in dfs.items()}


    levels = ['easy', 'moderate', 'hard']
    splits = ['train', 'test', 'valid']

    # Initialize plot_data
    plot_data = {level: {'count': []} for level in levels}

    for level in levels:
        level_data = {split: 0 for split in splits}
        total_images = 0

        for split in splits:
            split_key = f'{split}_{level}'
            count = image_counts.get(split_key, 0)
            level_data[split] = count
            total_images += count

        for split in splits:
            plot_data[level]['count'].append(level_data[split])


    plot_df = pd.DataFrame(plot_data)

    # Create a single plot with 3 subplots
    fig, axs = plt.subplots(len(levels), 1, figsize=(14, 12), sharex='col', sharey='row')

    # Plot data
    for i, level in enumerate(levels):
        axs[i].bar(splits, plot_df[level]['count'], color='skyblue')
        axs[i].set_title(f'Number of Images - {level.capitalize()}')
        axs[i].set_ylabel('Number of Images')

    # Set common x-axis label
    for ax in axs[-1:]:
        ax.set_xlabel('Split')

    plt.tight_layout()
    plt.show()


def plot_ovelap():
    """
    Plot the overlap of images in the dataset splits.
    :param dfs: Dictionary of pandas DataFrames for each dataset split
    :return: None
    """
    # Define sets for easier intersection calculation
    sets = {key: set(dfs[key]['image'].unique()) for key in dfs.keys()}

    # Calculate overlap percentages
    overlap_df = pd.DataFrame(index=sets.keys(), columns=sets.keys())

    for key1 in sets.keys():
        for key2 in sets.keys():
            if key1 == key2:
                overlap_df.loc[key1, key2] = 100.0
            else:
                overlap = len(sets[key1] & sets[key2])
                total = len(sets[key1] | sets[key2])
                overlap_df.loc[key1, key2] = (overlap / total) * 100

    plt.figure(figsize=(12, 8))
    sns.heatmap(overlap_df.astype(float), cmap='viridis', annot=False, fmt=".1f", vmin=0, vmax=100)
    plt.title('Percentage of Image Overlap between Splits')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


if __name__ == '__main__':
    plot_distributions()
    plot_ovelap()
