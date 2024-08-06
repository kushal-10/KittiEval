from ultralytics import YOLOv10
import os
from clearml import Task
task = Task.init(project_name='ida-ml', task_name='yolov10x-extreme-vanilla-a100')

# Load a model
# Training from scratch
model = YOLOv10("yolov10x.yaml")

# Train the model
dataset_path = os.path.join('data.yaml')
results = model.train(data=dataset_path, epochs=100, batch=4, imgsz=640, num_workers=4,
                      save=True, device=0, cache=True, project='trained_models',
                      name='yolo10n_extreme_vanilla_a100', pretrained=False, plots=True)

# Save to HF
HF_TOKEN = os.getenv('HF_TOKEN')
model.push_to_hub("Koshti10/yolov10x-trained-Kitti-2D-detection", token=HF_TOKEN)
