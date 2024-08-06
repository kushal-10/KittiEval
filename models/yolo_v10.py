from ultralytics import YOLOv10
import pandas as pd
import os
import re
from tqdm import tqdm
import json
import argparse
import numpy as np

BASE_DIR = 'splits'
SUB_DIR = 'difficulty_splits'

class YoloInference:
    """
    Inference model for YOLOv10. Initialize with model_name
    """
    def __init__(self, model_name: str, difficulty_type: str,  level: str, split: str):
        self.model_name = model_name
        self.difficulty_type = difficulty_type
        self.level = level
        self.split = split
        if model_name.endswith('.pt'):
            self.model = YOLOv10(model_name)
            self.class_id = 0
        else:
            self.model = YOLOv10.from_pretrained(model_name)
            self.class_id = 2

        split_path = os.path.join(BASE_DIR, SUB_DIR, difficulty_type, level, split+".json")

        with open(split_path, 'r') as file:
            self.data = json.load(file)

    def infer(self, image_path):
        """
        Get predictions of bounding boxes from image
        :param image_path: Path to image
        :return: predictions with bounding boxes of cars and speed of inference
        """

        predictions = {'bboxes': [], 'speed': {}, 'conf': []}

        # Run inference on a list of images
        results = self.model([image_path])  # return a list of Results objects

        # Process results list
        for result in results:
            boxes = result.boxes.xyxy.cpu()  # Boxes object for bounding box outputs
            classes = result.boxes.cls.cpu()  # Class values - integers
            speed = result.speed  # preprocess, inference, postprocess
            conf = result.boxes.conf.cpu().numpy()  # Confidence score for each box
            for i, box in enumerate(boxes):
                if int(classes[i]) == self.class_id:  # Consider only Car predictions
                    predictions['bboxes'].append(box.cpu().numpy())

            predictions['speed'] = speed
            predictions['conf'] = conf

        return predictions

    def generate_result(self):
        """
        Generate predictions for the images on selected split and save under results
        :return: None
        """

        RES_DIR = os.path.join('results', self.difficulty_type, self.level, self.split)
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)

        json_data = []
        for key in tqdm(self.data, desc=f"Generating Predictions for {self.model_name}"):
            ground_truths = self.data[key]
            pred = self.infer(key)

            pred_bboxes = pred['bboxes']
            pred_conf = pred['conf']
            for i in range(len(pred_bboxes)):
                curr_pred = pred_bboxes[i].tolist()
                curr_conf = float(pred_conf[i])
                curr_pred.append(curr_conf)  # append confidence score to the bbox values

                # Convert into proper format and append
                curr_data = {'Car_prediction': curr_pred}
                ground_truths.append(curr_data)

            # Add speed information as well - preprocess, inference and postprocess
            speed_data = {
                'speed_data': [
                    pred['speed']['preprocess'],
                    pred['speed']['inference'],
                    pred['speed']['postprocess']
                ]
            }
            ground_truths.append(speed_data)
            json_data_instance = {key: ground_truths}
            json_data.append(json_data_instance)

        if self.model_name.endswith('.pt'):
            name_splits = self.model_name.split('-')
        else:
            name_splits = self.model_name.split('/')
        save_path = os.path.join(RES_DIR, f"{name_splits[0]}_{name_splits[1]}.json")

        with open(save_path, 'w') as file:
            json.dump(json_data, file, indent=4)
        print("Predictions saved to {}".format(save_path))


def save_results(model_name, diff_type, level, split):
    InferModel = YoloInference(model_name, diff_type, level, split)
    InferModel.generate_result()


if __name__ == '__main__':

    # python3 models/yolo_v10.py --model_name jameslahm/yolov10x --split test --level easy --type custom

    parser = argparse.ArgumentParser(description='Run YOLO inference with specified parameters.')
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to use.')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'valid'], required=True,
                        help='The dataset split to use.')
    parser.add_argument('--level', type=str, choices=['hard', 'easy', 'moderate', 'extreme'], required=True,
                        help='The difficulty level of the dataset.')
    parser.add_argument('--type', type=str, choices=['custom', 'base'], required=True,
                        help='The type of the dataset.')

    args = parser.parse_args()

    save_results(args.model_name, args.type, args.level, args.split)
