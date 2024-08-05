from ultralytics import YOLOv10
import os
from clearml import Task
task = Task.init(project_name='ida-ml', task_name='yolov10x-extreme-vanilla')

# Load a model
# Training from scratch
model = YOLOv10("yolov10x.yaml")

# Train the model
dataset_path = os.path.join('data.yaml')
results = model.train(data=dataset_path, epochs=500, batch=256, imgsz=640)

# Save to HF
HF_TOKEN = os.getenv('HF_TOKEN')
model.push_to_hub("Koshti10/yolov10x-trained-Kitti-2D-detection", token=HF_TOKEN)
