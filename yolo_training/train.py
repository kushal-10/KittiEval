from ultralytics import YOLOv10
import os
from clearml import Task
import argparse


def train_model(model_name: str = "yolov10n", mode: str = 'vanilla', batch_size: int = 16, gpu_name: str = 'a100', multi_gpu: str = '1'):
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

    dataset_path = os.path.join('data.yaml')

    # Convert str to int
    batch_size = int(batch_size)

    # Setup appropriate GPU devices
    if multi_gpu == '1':
        multi_gpu = [0]
    elif multi_gpu == '4':
        multi_gpu = [0,1,2,3]
    else:
        multi_gpu = [0,1,2,3,4,5,6,7]

    # Get HF token
    HF_TOKEN = os.getenv('HF_TOKEN')

    # Load a model
    if mode == 'vanilla':
        # Training from scratch
        model = YOLOv10(f"{model_name}.yaml")
        results = model.train(data=dataset_path, epochs=100, batch=batch_size, imgsz=640, workers=1,
                              save=True, device=multi_gpu, cache=True, project='trained_models',
                              name=f'{model_name}_extreme_{mode}_{gpu_name}', pretrained=False, plots=True)
        # Save to HF
        model.push_to_hub(f"Koshti10/{model_name}-trained-Kitti-2D-detection", token=HF_TOKEN)

    else:
        model = YOLOv10.from_pretrained(f"jameslahm/{model_name}")
        results = model.train(data=dataset_path, epochs=100, batch=batch_size, imgsz=640, workers=1,
                              save=True, device=multi_gpu, cache=True, project='trained_models',
                              name=f'{model_name}_extreme_{mode}_{gpu_name}', pretrained=True, plots=True)

        # Save to HF
        model.push_to_hub(f"Koshti10/{model_name}-finetuned-Kitti-2D-detection", token=HF_TOKEN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('model_name', choices=['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x'],
                        help='The name of the model to train. Options are yolov10n/s/m/b/l/x.')
    parser.add_argument('mode', choices=['pt', 'vanilla'],
                        help='The mode to use for training. Options are pt or vanilla.')
    parser.add_argument('batch_size', choices=['16', '32', '64', '128', '256', '512'],
                        help='The batch size to use for training.')
    parser.add_argument('gpu_name', choices=['a100', '1080'],
                        help='The gpu to use for training. Options are a100 or 1080')
    parser.add_argument('multi_gpu', choices=['1', '4', '8'],
                        help='The number of gpus to use for training. Options are size of list - 1,4,8')

    args = parser.parse_args()
    train_model(args.model_name, args.mode, args.batch_size, args.gpu_name, args.multi_gpu)
