from ultralytics import YOLOv10
import os

# Load a model
# Training from scratch
model = YOLOv10()

# Train the model
dataset_path = os.path.join('data', 'huggingface', 'data.yaml')
results = model.train(data=dataset_path, epochs=500, batch=256, imgsz=640)

# Save to HF
model.push_to_hub("Koshti10/yolov10-trained-Kitti-2D-detection", token="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")
