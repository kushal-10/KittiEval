## Requires {train/valid/test}_split.csv files under /dataset/
import pandas as pd
import os

# Define difficulty criteria
def classify_difficulty(df):
    """
    :param df: Split df
    :return: df splits by difficulty category
    """

    columns = ['image', 'bbox']
    easy_df = pd.DataFrame(columns=columns)
    moderate_df = pd.DataFrame(columns=columns)
    hard_df = pd.DataFrame(columns=columns)

    # Initialize row lists
    easy_rows = []
    moderate_rows = []
    hard_rows = []

    for i in range(len(df)):
        row = df.iloc[i]
        top = row['left-top-y']
        bottom = row['right-bottom-y']
        height = abs(top - bottom)

        occlusion = row['occluded']
        truncation = row['truncated']
        image = row['image']

        row_item = {'image': image,
                    'bbox': [row['left-top-x'], row['left-top-y'],
                             row['right-bottom-x'], row['right-bottom-y']]}

        ## Same image and bbox can be used in multiple difficulties
        if height >= 40 and occlusion <= 0 and truncation <= 0.15:
            easy_rows.append(row_item)

        if height >= 25 and occlusion <= 1 and truncation <= 0.3:
            moderate_rows.append(row_item)

        if height >= 25 and occlusion <= 2 and truncation <= 0.5:
            hard_rows.append(row_item)

    temp_easy_df = pd.DataFrame(easy_rows, columns=columns)
    temp_moderate_df = pd.DataFrame(moderate_rows, columns=columns)
    temp_hard_df = pd.DataFrame(hard_rows, columns=columns)

    easy_df = pd.concat([easy_df, temp_easy_df], ignore_index=True)
    moderate_df = pd.concat([moderate_df, temp_moderate_df], ignore_index=True)
    hard_df = pd.concat([hard_df, temp_hard_df], ignore_index=True)

    return easy_df, moderate_df, hard_df


if __name__ == '__main__':

    splits = ['train', 'valid', 'test']

    for s in splits:
        df = pd.read_csv(os.path.join('dataset', 'csvs', s+'_split.csv'))
        easy, moderate, hard = classify_difficulty(df)
        # save dir
        SAVE_DIR = os.path.join('dataset', 'csvs')
        easy_save_path = os.path.join(SAVE_DIR, s+'_easy_split.csv')
        moderate_save_path = os.path.join(SAVE_DIR, s+'_moderate_split.csv')
        hard_save_path = os.path.join(SAVE_DIR, s+'_hard_split.csv')

        # Save to CSV files
        easy.to_csv(easy_save_path, index=False)
        moderate.to_csv(moderate_save_path, index=False)
        hard.to_csv(hard_save_path, index=False)

        print(f"Length of easy: {len(easy)}, moderate: {len(moderate)}, hard: {len(hard)} for {s} split")

    print("Files saved under /dataset/csvs/")
