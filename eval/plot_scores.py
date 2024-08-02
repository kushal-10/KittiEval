import os
import matplotlib.pyplot as plt
import numpy as np

from eval.score_yolo import YOLOScorer

RES_DIR = 'results'
dirs = os.listdir(RES_DIR)

# Collect result JSONs
json_paths = []
for dataset in dirs:
    path = os.path.join(RES_DIR, dataset)
    levels = os.listdir(path)
    for level in levels:
        split_path = os.path.join(path, level)
        for split in os.listdir(split_path):
            jsons = os.listdir(os.path.join(split_path, split))
            for json in jsons:
                json_paths.append(os.path.join(split_path, split, json))


def plot_pr_curve(json_path: str):
    """
    :param json_path: Path to json file
    :return:
    """

    thresholds = np.linspace(0, 1, 41)
    precisions = []
    recalls = []

    for t in thresholds:
        scorer = YOLOScorer(json_path, 0.7, t, 40)
        p, r = scorer.get_precision_recall()
        precisions.append(p)
        recalls.append(r)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

    return precisions, recalls


def calculate_ap(precisions, recalls):
    # Convert lists to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Append a point at (0,1) to the beginning to cover the edge case
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([1.0], precisions))

    # Sort precision and recall values
    indices = np.argsort(recalls)
    recalls = recalls[indices]
    precisions = precisions[indices]

    # Ensure precision is non-increasing
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Compute the Average Precision using the trapezoidal rule
    ap = np.sum(np.diff(recalls) * precisions[:-1])

    return ap


if __name__ == '__main__':
    path = json_paths[0]
    p, r = plot_pr_curve(path)
    ap = calculate_ap(p, r)
    print(f"Average Precision: {ap}")