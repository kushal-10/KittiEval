import json
import os

def process_results(json_path: str):
    """
    :param json_path: Path to the json file containing the results
    :return: processed results in a proper form
    """

    with open(json_path, 'r') as f:
        image_data = json.load(f)

    processed_res = []
    for data in image_data:
        data_object = {
            "Car": [],
            "Car_prediction": [],
            "DontCare": [],
            "Speeds": None
        }
        for key in list(data.keys()):
            image_objects = data[key]
            for image_object in image_objects:
                for k in list(image_object.keys()):
                    if k == "Car":
                        data_object["Car"].append(image_object[k])
                    elif k == "Car_prediction":
                        data_object["Car_prediction"].append(image_object[k])
                    elif k == "DontCare":
                        data_object["DontCare"].append(image_object[k])
                    else:
                        data_object["Speeds"] = image_object[k]

        processed_res.append(data_object)

    return processed_res


class YOLOScorer:
    """
    Class to evaluate YOLO type models,
    Get Precision and Recall for a predefined IOU and Confidence threshold
    """

    def __init__(self, result_path, iou_threshold: float = 0.7, conf_threshold: float = 0.5):
        self.result_path = result_path
        self.iou_threshold = iou_threshold
        self. conf_threshold = conf_threshold

    @staticmethod
    def calculate_iou(prediction: list, ground_truth: list):
        """
        :param prediction: The predicted BBox
        :param ground_truth: THe Ground Truth BBox
        :return: IOU score
        """

        left = max(prediction[0], ground_truth[0])  # Left most point of overlap
        right = min(prediction[2], ground_truth[2])  # Right most point of overlap
        if right <= left:
            width = 0
        else:
            width = right - left

        top = max(prediction[1], ground_truth[1])  # Top most point of overlap
        bottom = max(prediction[3], ground_truth[3])  # Bottom most point of overlap
        if bottom <= top:
            height = 0
        else:
            height = bottom - top

        # Calculate IOU score
        area_overlap = width * height
        area_pred = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
        area_gt = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])

        iou_score = area_overlap / (area_pred + area_gt - area_overlap)

        return iou_score

    def counts(self, data_object: dict, iou_threshold: float = 0.7, conf_threshold: float = 0.5):
        """
        Calculate counts of FP, TP, and FN for a given image predictions

        :param data_object: Prediction data for a given image, may contain "Car", "Car_prediction", "DontCare"
        :param iou_threshold: The threshold of IOU between GT and Pred
        :param conf_threshold: The threshold of confidence of the prediction
        :return: Counts for TP, FP, FN
        """

        predictions = data_object["Car_prediction"]
        ground_truths = data_object["Car"]
        dont_cares = data_object["DontCare"]

        # Base cases
        if not ground_truths and not predictions:
            return 0, 0, 0
        elif ground_truths and not predictions:
            return 0, 0, len(ground_truths)
        elif predictions and not ground_truths:
            return 0, len(predictions), 0
        else:

            matched_gt = set()
            matched_dc = set()
            TP = 0
            FP = 0
            FN = 0
            for pred in predictions:
                # Consider only the predictions for a given threshold of confidence score
                # Used for AP calculation
                # Discard the rest of the predictions
                if pred[-1] >= conf_threshold:
                    match_gt = []
                    for gt in ground_truths:
                        iou_score = self.calculate_iou(pred, gt)
                        match_gt.append([iou_score, gt])
                    sorted_match_gt = sorted(match_gt, reverse=True)

                    # Check the maximum matching Ground truth bbox for this prediction
                    # If a bbox that was not matched previously is matched with this prediction, inc TP
                    if sorted_match_gt[0][0] >= iou_threshold:
                        if tuple(sorted_match_gt[0][1]) not in matched_gt:
                            TP += 1
                            matched_gt.add(tuple(sorted_match_gt[0][1]))
                            break

                    # Alt case - This predicted BBox was not matched with any GT box
                    # Check if any overlap with dontcare Region; If yes discard this, else inc FP count
                    # Ref Eval - https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
                    # Check this only if Dont cares exist in the Ground Truths
                    if dont_cares:
                        match_dc = []
                        for dc in dont_cares:
                            iou_score = self.calculate_iou(pred, dc)
                            match_dc.append([iou_score, dc])
                        sorted_match_dc = sorted(match_dc, reverse=True)

                        if sorted_match_dc[0][0] >= iou_threshold:
                            if tuple(sorted_match_dc[0][1]) not in matched_dc:
                                matched_dc.add(tuple(sorted_match_dc[0][1]))  # Add  to matched_dc and skip this prediction
                                break

                    # If none of the above two cases are valid, add to False positive count
                    FP += 1

            # Done with TPs and FPs, calculate FNs - Object there, but MF was not detected
            FN = len(ground_truths) - len(matched_gt)

            return TP, FP, FN

    def get_pr(self, processed_results, iou_threshold: float = 0.7, conf_threshold: float = 0.5):
        """
        Get Precision and Recall
        :param processed_results: A list of predictions for all images in a given setting
        :param iou_threshold: Threshold for calculating IOU scores
        :param conf_threshold: Threshold for discarding predictions
        :return: P, R - Precision and Recall at a given IoU threshold and confidence threshold
        """

        TP = 0
        FP = 0
        FN = 0
        for res in processed_results:
            tp, fp, fn = self.counts(res, iou_threshold, conf_threshold)
            TP += tp
            FP += fp
            FN += fn

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)

        return Precision, Recall

    def get_precision_recall(self):
        """
        :return: Precision and Recall for a given result file
        """
        processed_results = process_results(self.result_path)
        precision, recall = self.get_pr(processed_results, self.iou_threshold, self.conf_threshold)
        return precision, recall


if __name__ == '__main__':

    results_path = os.path.join("results", "easy", "jameslahm_yolov10x_test.json")
    ious = [0.1, 0.2, 0.3, 0.4, 0.9, 1]
    confs = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]

    for i in ious:
        for c in confs:
            yolo_scorer = YOLOScorer(results_path, i, c)
            precision, recall = yolo_scorer.get_precision_recall()
            print(f"Precision : {precision}, Recall : {recall} for IOU Threshold: {i} and Conf: {c}")


