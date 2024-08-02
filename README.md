# KittiEval
Project for SoSe24 course - Intelligent Data Analysis and Machine Learning II

## Setup

1) Clone this repository

```
git clone https://github.com/kushal-10/KittiEval.git
cd KittiEval
source prepare_path.sh
```

2) Download the Kitti Dataset available [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Store the images under `ROOT/data/images` and labels under `ROOT/data/labels`.


## Create Splits

1) Collect the labels from text files. This will create a CSV file of labels - `splits/csvs/labels.csv` 
```
python3 splits/collect_labels.py
```

2) Create Train, Test and Validation splits. This will create 3 files - `train_split.csv`, `test_split.csv` and `valid_split.csv`
```
python3 splits/create_splits.py
```

3) Create splits based on difficulty level


## Inference

1) For base YoLo models - Run the following command with the required args to generate predictions. This will save 
a CSV file containing ground truth values and predictions for an image along with the inference speeds under `results`

For example, to generate predictions on Yolo_v10x on test set of Easy level, run the following
```
python3 base_models/yolo_v10.py --model_name jameslahm/yolov10x --split test --level easy
```
- model_name (required): The name of the YOLO model to use. Example values: `jameslahm/yolov10x`, `jameslahm/yolov10s`, `jameslahm/yolov10n`

- split (required): The dataset split to use. Valid options are: `train`, `test`, `valid`

- level (required): The difficulty level of the dataset. Valid options are: `hard`, `easy`, `moderate`
