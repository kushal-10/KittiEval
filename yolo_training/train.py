from ultralytics import YOLOv10
import os
from clearml import Task
import argparse

def train_model(model_name: str = "yolov10n", mode: str = 'vanilla', batch_size: int = 16,
                multi_gpu: str = '1', freeze=None, lr0: float = 0.01, lrf: float = 0.01,
                momentum: float = 0.937, optimizer: str = 'auto'):

    """
    Train a model with specified parameters and save
    :param model_name: The name of the yolo model
    :param mode: Type of model to train: vanilla - form scratch, pt - from pretrained
    :param batch_size: Batch size for training
    :param multi_gpu: Pass a list of devices if multi-gpu setup is used else, pass empty
                      By default will use device 0 if available else cpu
    :param freeze: First N number of layers to freeze during training
    :param lr0: Initial learning rate
    :param lrf: Final learning rate
    :param momentum: Momentum
    :param optimizer: Type of optimizer
    """
    print(f"Training model {model_name} with mode {mode}")
    task = Task.init(project_name='ida-ml', task_name=f'{model_name}-{mode}-final')

    dataset_path = os.path.join('data.yaml')

    # Setup Parameters
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
    # Learning rates
    lr0 = float(lr0)
    lrf = float(lrf)
    momentum = float(momentum)
    # Freeze layers
    freeze = int(freeze)

    # Load a model
    if mode == 'vanilla':
        # Training from scratch
        model = YOLOv10(f"{model_name}.yaml")
        results = model.train(data=dataset_path, epochs=100, batch=batch_size, imgsz=640, workers=1,
                              save=True, device=multi_gpu, cache=True, project='final_models',
                              lr0=lr0, lrf=lrf, momentum=momentum, optimizer=optimizer, freeze=None,
                              pretrained=False, plots=True)
        # Save to HF
        model.push_to_hub(f"Koshti10/{model_name}_{mode}_tuned", token=HF_TOKEN)

    else:
        model = YOLOv10.from_pretrained(f"jameslahm/{model_name}")
        results = model.train(data=dataset_path, epochs=100, batch=batch_size, imgsz=640, workers=1,
                              save=True, device=multi_gpu, cache=True, project='final_models',
                              lr0=lr0, lrf=lrf, momentum=momentum, optimizer=optimizer, freeze=freeze,
                              pretrained=True, plots=True)

        # Save to HF
        model.push_to_hub(f"Koshti10/{model_name}_{mode}_tuned", token=HF_TOKEN)

def parse_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return ivalue

def parse_positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float.")
    return fvalue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('model_name', choices=['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x'],
                        help='The name of the model to train. Options are yolov10n/s/m/b/l/x.')
    parser.add_argument('mode', choices=['pt', 'vanilla'],
                        help='The mode to use for training. Options are pt or vanilla.')
    parser.add_argument('batch_size', type=parse_positive_int,
                        help='The batch size to use for training. Must be a positive integer.')
    parser.add_argument('multi_gpu', type=parse_positive_int,
                        help='The number of GPUs to use for training. Must be a positive integer 1 4 or 8.')
    parser.add_argument('freeze', type=parse_positive_int,
                        help='The number of layers to freeze for training. Must be a positive integer. Use 300 for nano and 600 for XL')
    parser.add_argument('lr0', type=parse_positive_float,
                        help='The initial learning rate. Must be a positive float.')
    parser.add_argument('lrf', type=parse_positive_float,
                        help='The final learning rate. Must be a positive float.')
    parser.add_argument('momentum', type=parse_positive_float,
                        help='The momentum value. Must be a positive float.')
    parser.add_argument('optimizer', choices=['auto', 'AdamW', 'SGD'],
                        help='The optimizer to use. Options are auto, AdamW, SGD.')

    args = parser.parse_args()

    train_model(args.model_name, args.mode, args.batch_size, args.multi_gpu, args.freeze,
                args.lr0, args.lrf, args.momentum, args.optimizer)
