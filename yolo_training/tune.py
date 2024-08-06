import numpy as np
from ultralytics import YOLOv10
import os
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective, plot_gaussian_process


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

    # Config
    multi_gpu = [0]  # Set according to GPU availability
    model_name = 'yolov10n'   # OR yolov10x
    mode = 'pt'   # OR 'vanilla'
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
    Integer(16, 32, name='batch_size'),
    Integer(25, 100, name='epochs'),
    Categorical(categories=['AdamW', 'SGD'], name='optimizer'),
    Real(1e-3, 1e-2, name='lr0'),
    Real(1e-3, 1e-2, name='lrf'),
    Real(0.8, 0.99, name='momentum')
]

def run_optimization():
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=50,  # Number of evaluations
        random_state=42
    )

    # Create and save plots
    # Ensure the directory exists
    output_dir = 'optimization_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Save convergence plot
    plt.figure()
    plot_convergence(result)
    plt.title('Convergence Plot')
    plt.savefig(os.path.join(output_dir, 'convergence_plot.png'))
    plt.close()

    # Save objective plot
    plt.figure()
    plot_objective(result)
    plt.title('Objective Function Plot')
    plt.savefig(os.path.join(output_dir, 'objective_plot.png'))
    plt.close()

    # Save gaussian process
    plt.figure()
    plot_gaussian_process(result)
    plt.title('Gaussian Process')
    plt.savefig(os.path.join(output_dir, 'gaussian_plot.png'))
    plt.close()

    # Print the best hyperparameters and score
    print("Best hyperparameters found:")
    print("Batch size:", result.x[0])
    print("Epochs:", result.x[1])
    print("Image size:", result.x[2])
    print("Multi-GPU mode:", result.x[3])
    print("Best score (negative mAP50-95):", result.fun)


if __name__ == '__main__':
    run_optimization()

