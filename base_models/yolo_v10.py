from ultralytics import YOLOv10
import pandas as pd
import os
import re
from tqdm import tqdm
from contextlib import redirect_stdout
import argparse


def create_metadata(split_path: str):
    """
    Create metadata containing a single image entry with gold labels
    :param split_path: Path of split under consideration
    :return:
    """

    df = pd.read_csv(split_path)
    df['bbox'] = df['bbox'].apply(parse_bbox)

    image_bboxes = {}
    for i in range(len(df)):
        # Get image number
        image = df['image'].iloc[i]

        # Create image path
        image_str = str(image)
        while len(image_str) != 6:
            image_str = "0"+image_str
        image_str += ".png"
        image_path = os.path.join('data', 'images', image_str)

        if image_path not in image_bboxes:
            image_bboxes[image_path] = [df['bbox'].iloc[i]]
        else:
            image_bboxes[image_path].append(df['bbox'].iloc[i])

    return image_bboxes

def parse_bbox(bbox_str):
    bbox_str = bbox_str.replace('np.float64(', '').replace(')', '')
    bbox_list = re.findall(r"([0-9.]+)", bbox_str)
    # Convert to floats
    return [float(x) for x in bbox_list]


class YoloInference:
    """
    Inference model for YOLOv10. Initialize with model_name
    """
    def __init__(self, model_name: str, split: str, level: str):
        self.model_name = model_name
        self.split = split
        self.level = level
        self.model = YOLOv10.from_pretrained(model_name)
        split_path = os.path.join('dataset', 'csvs', split+"_"+level+"_split.csv")
        self.metadata = create_metadata(split_path)

    def infer(self, image_path):
        """
        Get predictions of bounding boxes from image
        :param image_path: Path to image
        :return: predictions with bounding boxes of cars and speed of inference
        """

        predictions = {'bboxes': [], 'speed': {}}

        # Run inference on a list of images
        results = self.model([image_path])  # return a list of Results objects

        # Process results list
        for result in results:
            boxes = result.boxes.xyxy.cpu()  # Boxes object for bounding box outputs
            classes = result.boxes.cls.cpu()
            speed = result.speed  # preprocess, inference, postprocess
            for i, box in enumerate(boxes):
                if int(classes[i]) == 2:
                    predictions['bboxes'].append(box.cpu().numpy())

            predictions['speed'] = speed

        return predictions


    def generate_result(self):
        """
        Generate predictions for the images on selected split and save under results
        :return: None
        """

        RES_DIR = os.path.join('results', self.level)
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)

        metadata_json = self.metadata

        images = []
        ground_truths = []
        predictions = []
        preprocess_speed = []
        inference_speed = []
        postprocess_speed = []

        for key in tqdm(metadata_json, desc=f"Generating Predictions for {self.model_name}"):
            images.append(key)
            gold_bboxes = metadata_json[key]
            ground_truths.append(gold_bboxes)

            pred = self.infer(key)
            predictions.append(pred['bboxes'])
            preprocess_speed.append(pred['speed']['preprocess'])
            inference_speed.append(pred['speed']['inference'])
            postprocess_speed.append(pred['speed']['postprocess'])


        result_data = {
            'image': images,
            'ground_truth': ground_truths,
            'prediction': predictions,
            'preprocess_speed': preprocess_speed,
            'inference_speed': inference_speed,
            'postprocess_speed': postprocess_speed
        }

        name_splits = self.model_name.split('/')
        save_path = os.path.join(RES_DIR, f"{name_splits[0]}_{name_splits[1]}_{self.split}.csv")

        result_df = pd.DataFrame.from_dict(result_data)
        result_df.to_csv(save_path, index=False)
        print("Predictions saved to {}".format(save_path))


def save_results(model_name, split, level):
    InferModel = YoloInference(model_name, split, level)
    InferModel.generate_result()


if __name__ == '__main__':

    # python3 base_models/yolo_v10.py --model_name jameslahm/yolov10x --split test --level easy

    parser = argparse.ArgumentParser(description='Run YOLO inference with specified parameters.')
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to use.')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'valid'], required=True,
                        help='The dataset split to use.')
    parser.add_argument('--level', type=str, choices=['hard', 'easy', 'moderate'], required=True,
                        help='The difficulty level of the dataset.')

    args = parser.parse_args()

    save_results(args.model_name, args.split, args.level)
