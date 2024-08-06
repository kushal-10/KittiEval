from ultralytics import YOLOv10
import os
from clearml import Task
import argparse


def train_model(model_name: str = "yolov10n", mode: str = 'vanilla', batch_size: int = 16, gpu_name: str = 'a100', multi_gpu: list = []):
    """
    Train a model with specified parameters
    :param model_name: The name of the yolo model
    :param mode: Type of model to train: vanilla - form scratch, pt - from pretrained
    :param batch_size: Batch size for training
    :param gpu: Type of GPU to train on - Only used for saving checkpoints
    :param multi_gpu: Pass a list of devices if multi-gpu setup is used else, pass empty
                      By default will use device 0 if available else cpu
    """
    print(f"Training model {model_name} with mode {mode}")
    task = Task.init(project_name='ida-ml', task_name=f'{model_name}-extreme-{mode}-{gpu_name}')

    # Load a model
    # Training from scratch
    model = YOLOv10(f"{model_name}.yaml")

    # Train the model
    dataset_path = os.path.join('data.yaml')
    results = model.train(data=dataset_path, epochs=100, batch=batch_size, imgsz=640, workers=1,
                          save=True, device=multi_gpu, cache=True, project='trained_models',
                          name=f'{model_name}_extreme_{mode}_{gpu_name}', pretrained=False, plots=True)

    # Save to HF
    HF_TOKEN = os.getenv('HF_TOKEN')
    model.push_to_hub(f"Koshti10/{model_name}-trained-Kitti-2D-detection", token=HF_TOKEN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('model_name', choices=['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x'],
                        help='The name of the model to train. Options are yolov10n/s/m/b/l/x.')
    parser.add_argument('mode', choices=['pt', 'vanilla'],
                        help='The mode to use for training. Options are pt or vanilla.')
    parser.add_argument('batch_size', choices=[2**n],
                        help='The batch size to use for training.')
    parser.add_argument('gpu_name', choices=['a100', '1080'],
                        help='The gpu to use for training. Options are a100 or 1080')
    parser.add_argument('multi_gpu', choices=[list[0:n]],
                        help='The number of gpus to use for training. Option is a list defining number [0...n]')

    args = parser.parse_args()
    train_model(args.model_name, args.mode, args.batch_size, args.gpu_name, args.multi_gpu)
