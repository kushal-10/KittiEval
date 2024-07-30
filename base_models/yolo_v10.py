from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10n')

# Run batched inference on a list of images
results = model(["data/images/000000.png", "data/images/000036.png"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
    print(boxes)
