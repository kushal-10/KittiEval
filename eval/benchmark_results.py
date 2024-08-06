import pandas as pd
from eval.score_yolo import YOLOScorer

from tqdm import tqdm
import numpy as np
import os

RES_DIR = "results"

class BenchmarkResults:
    def __init__(self):
        pass

    @staticmethod
    def get_result_data(res_dir: str = RES_DIR):
        """

        :param res_dir: The root directory containing the results
        :return: A list of Result objects, containing result_file paths, levels and dataset type
        """
        res_data = []

        datasets = os.listdir(res_dir)

        for dataset in datasets:
            if os.path.isdir(os.path.join(res_dir, dataset)):
                levels = os.listdir(os.path.join(res_dir, dataset))
                for level in levels:
                    splits = os.listdir(os.path.join(res_dir, dataset, level))
                    for split in splits:
                        jsons = os.listdir(os.path.join(res_dir, dataset, level, split))
                        for json in jsons:
                            model_name = json.split('.')[0]
                            json_object = {
                                "file_path": os.path.join(res_dir, dataset, level, split, json),
                                "level": level,
                                "dataset": dataset,
                                "model_name": model_name
                            }
                            res_data.append(json_object)

        return res_data

    @staticmethod
    def get_scorer_args(res_obj):
        """
        :param res_obj: Result object containing file path, dataset type and difficulty level
        :return: Args -> min_height for the scorer to consider/discard predictions
        """
        if res_obj["dataset"] == "custom":
            if res_obj["level"] == 'easy':
                return 40
            elif res_obj["level"] == 'moderate':
                return 30
            elif res_obj["level"] == 'hard':
                return 25
            else:
                return 0

        # if res_obj["dataset"] == "base":
        #     if res_obj["level"] == 'extreme':
        #         return 0
        #     else:
        #         return -1

        # Consider Base as well
        # The performance of base_easy should be worse than custom_easy
        # More False Positives (Not actually,...but ok)
        if res_obj["dataset"] == "base":
            if res_obj["level"] == 'easy':
                return 40
            elif res_obj["level"] == 'moderate':
                return 30
            elif res_obj["level"] == 'hard':
                return 25
            else:
                return 0

    @staticmethod
    def get_ap_score(precisions, recalls):
        """
        :param precisions: A list of precisions
        :param recalls: A list of recalls

        :return: Average Precision score
        """
        precisions = np.array(precisions)
        recalls = np.array(recalls)

        # Append a point at (0,1) to the beginning to cover the edge case
        # recalls = np.concatenate(([0.0], recalls))
        # precisions = np.concatenate(([1.0], precisions))

        # Sort precision and recall values
        indices = np.argsort(recalls)
        recalls = recalls[indices]
        precisions = precisions[indices]

        # Ensure precision is non-increasing
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]

        # Compute the Average Precision using the trapezoidal rule
        # ap = np.sum(np.diff(recalls) * precisions[:-1])

        # Try with 11 point AP
        # Fixed recall levels for 11-point AP
        recall_levels = np.linspace(0, 1, 41)
        recall_levels = recall_levels[1:]

        # Interpolate precision values at each recall level
        ap11_precisions = []
        for r in recall_levels:
            # Find the maximum precision for recall >= r
            relevant_precisions = precisions[recalls >= r]
            if relevant_precisions.size > 0:
                max_precision = np.max(relevant_precisions)
            else:
                max_precision = 0  # Or handle the case as needed
            ap11_precisions.append(max_precision)

        ap11 = np.mean(ap11_precisions)

        return ap11

    def calculate_ap_scores(self, file_path: str, min_height: int):
        """
        :param file_path: The result file path
        :param min_height: Minimum height in pixels
        :return: AP score
        """
        spacing = 100
        thresholds = np.linspace(0, 1, spacing)
        precisions = []
        recalls = []
        for t in thresholds:
            scorer = YOLOScorer(file_path, 0.7, t, min_height)
            p, r = scorer.get_precision_recall()
            precisions.append(p)
            recalls.append(r)
        ap_score = self.get_ap_score(precisions, recalls)

        return ap_score

    def generate_scores(self):
        """
        Generate the scores for each file in the collected JSON result files
        :return: A DF containing the scores for each file in the collected JSON result files
        """
        aps = []
        setting = []
        model = []

        res_data = self.get_result_data()
        for obj in tqdm(res_data, desc='Generating scores for all files'):
            file_path = obj['file_path']
            min_height = self.get_scorer_args(obj)
            setting.append(obj["level"] + "_" + obj["dataset"])
            model.append(obj["model_name"])

            if min_height == -1:
                aps.append(None)
            else:
                # Calculate AP with 11 threshold points
                ap_score = self.calculate_ap_scores(file_path, min_height)
                aps.append(ap_score)

        data = {
            "Model": model,
            "Dataset": setting,
            "APScore": aps
        }

        csv_save_path = os.path.join(RES_DIR, "results.csv")
        html_save_path = os.path.join(RES_DIR, "results.html")

        df = pd.DataFrame(data)
        df.to_csv(csv_save_path, index=False)
        df.to_html(html_save_path, index=False)
        print(f"Result Table saved to {csv_save_path} and {html_save_path}")


if __name__ == '__main__':
    res = BenchmarkResults()
    res.generate_scores()
