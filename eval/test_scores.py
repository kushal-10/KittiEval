import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box


def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    b1 = box(*box1)
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0


def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """Calculate precision and recall for given predictions and ground truths."""
    tp = 0
    fp = 0
    fn = 0

    matched_gt_indices = set()

    for pred in predictions:
        pred_box = pred[:4]
        pred_confidence = pred[4]
        best_iou = 0
        best_gt_index = -1

        for i, gt in enumerate(ground_truths):
            if i in matched_gt_indices:
                continue
            gt_box = gt
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = i

        if best_iou >= iou_threshold:
            if best_gt_index not in matched_gt_indices:
                tp += 1
                matched_gt_indices.add(best_gt_index)
            else:
                fp += 1
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt_indices)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return precision, recall


def average_precision(predictions, ground_truths):
    """Calculate Average Precision (AP)."""
    sorted_predictions = sorted(predictions, key=lambda x: x[4], reverse=True)

    precisions = []
    recalls = []
    previous_recall = 0

    for i, pred in enumerate(sorted_predictions):
        precision, recall = calculate_precision_recall(sorted_predictions[:i + 1], ground_truths)
        precisions.append(precision)
        recalls.append(recall)

    # Compute AP using the trapezoidal rule
    ap = 0
    for i in range(1, len(recalls)):
        delta_recall = recalls[i] - recalls[i - 1]
        ap += precisions[i] * delta_recall

    return ap


def process_json_file(json_path):
    """Process the JSON file to compute AP for each image and aggregate results."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_ground_truths = []
    all_predictions = []

    for entry in data:
        # Extract ground truths and predictions for each image
        image_path = list(entry.keys())[0]
        print(image_path)
        objects = entry[image_path]

        ground_truths = [obj["Car"] for obj in objects if "Car" in obj]
        predictions = [obj["Car_prediction"] for obj in objects if "Car_prediction" in obj]

        # Debugging output
        # print(f"Image: {image_path}")
        print(f"Ground Truths: {ground_truths}")
        print(f"Predictions: {predictions}")

        all_ground_truths.extend(ground_truths)
        all_predictions.extend(predictions)

    # Compute Average Precision for the entire dataset
    ap = average_precision(all_predictions, all_ground_truths)
    print(f"Average Precision (overall): {ap:.4f}")

    # Optional: Plot Precision-Recall curve for the dataset
    sorted_predictions = sorted(all_predictions, key=lambda x: x[4], reverse=True)
    precisions = []
    recalls = []

    for i in range(1, len(sorted_predictions) + 1):
        precision, recall = calculate_precision_recall(sorted_predictions[:i], all_ground_truths)
        precisions.append(precision)
        recalls.append(recall)

    plt.figure()
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.grid(True)
    plt.show()


# Path to your JSON file
json_path = 'results/easy/jameslahm_yolov10x_test.json'
process_json_file(json_path)
