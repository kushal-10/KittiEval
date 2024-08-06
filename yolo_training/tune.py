import numpy as np
from ultralytics import YOLOv10
import os
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from clearml import Task

def objective(params):
    """
    :param params: Hyperparameters
    :return:
    """
    batch_size, epochs, optimizer, lr0, lrf, momentum = params
    # Ensure that all parameters are converted to Python natives
    batch_size = int(batch_size)
    epochs = int(epochs)
    lr0 = float(lr0)
    lrf = float(lrf)
    momentum = float(momentum)

    task = Task.init(project_name='ida-ml', task_name=f'{batch_size}_{epochs}_{lr0}_{lrf}_{momentum}')

    # Config
    multi_gpu = [0,1,2,3]  # Set according to GPU availability
    model_name = 'yolov10n'   # OR yolov10x
    mode = 'vanilla'   # OR 'pt'
    dataset_path = os.path.join('data.yaml')

    # Training process
    if mode == 'vanilla':
        model = YOLOv10(f"{model_name}.yaml")
    else:
        model = YOLOv10.from_pretrained(f"jameslahm/{model_name}")

    results = model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        workers=1,
        save=True,
        device=multi_gpu,
        cache=True,
        project='tuned_models',
        pretrained=(mode == 'pt'),
        plots=True
    )

    # Extract the metric to be optimized (e.g., mAP50-95)
    metrics = results['results_dict']
    mAP50_95 = metrics.get('metrics/mAP50-95(B)', np.nan)

    # Return negative of the metric because gp_minimize minimizes the objective function
    return -mAP50_95


# Define the hyperparameter space
search_space = [
    Categorical([16,20,24,28,32], name='batch_size'),
    Categorical(categories=['AdamW', 'SGD', 'auto'], name='optimizer'),
    Real(1e-3, 1e-2, name='lr0'),
    Real(1e-3, 1e-2, name='lrf'),
    Real(0.8, 0.99, name='momentum')
]

def run_optimization():
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,  # Number of evaluations
        random_state=42
    )

    # Print the best hyperparameters and score
    print("Best hyperparameters found: ", result.x)
    print("Best score (negative mAP50-95):", result.fun)


if __name__ == '__main__':
    run_optimization()

