# KittiEval
Project for SoSe24 course - Intelligent Data Analysis and Machine Learning II

## 1) YOLO v10 Base Model Inference

#### A) Setup

1) Clone this repository

```
git clone https://github.com/kushal-10/KittiEval.git
cd KittiEval
source prepare_path.sh
```

2) Download the Kitti Dataset available [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Store the images under `ROOT/data/images` and labels under `ROOT/data/labels`.


#### B) Create Splits

##### All files are provided in the repo. This step can be skipped. 

1) Collect the labels from text files. This will create a CSV file of labels - `splits/csvs/labels.csv` 
```
python3 splits/collect_labels.py
```

2) Create Train, Test and Validation splits. This will create 3 files - `train_split.csv`, `test_split.csv` and `valid_split.csv`
```
python3 splits/create_splits.py
```

3) Create splits based on difficulty level. This creates all 3 splits for each difficulty level and each difficulty_type.
Added a custom difficulty type.
```
python3 splits/create_difficulty_splits.py
```


#### C) Inference

1) For base YoLo models - Run the following command with the required args to generate predictions. This will save 
a CSV file containing ground truth values and predictions for an image along with the inference speeds under `results`

For example, to generate predictions on Yolo_v10x on test set of Easy level, run the following
```
python3 base_models/yolo_v10.py --model_name jameslahm/yolov10x --split test --level easy
```
- model_name (required): The name of the YOLO model to use. Example values: `jameslahm/yolov10x`, `jameslahm/yolov10s`, `jameslahm/yolov10n`

- split (required): The dataset split to use. Valid options are: `train`, `test`, `valid`

- level (required): The difficulty level of the dataset. Valid options are: `hard`, `easy`, `moderate`

- type (required): The dataset type to use. Valid options are: `custom`, `base`

#### D) Evaluation

After Inference all result files will be saved hierarchically under results. Run the following to create `results.html`, and `results.csv`

```
python3 eval/benchmark_results.py
```

## 2) Training YOLO v10

The train, val set of `extreme` level is considered here from the previous evaluation. This covers all `Car` type predictions available in the dataset.

#### A) Creating custom dataset

The dataset first needs to be converted into YOLO format - Moving the images and labels to particular sub folders of the split.

```
python3 yolo_training/dataset/create_yolo_dataset.py
```

Run the following to setup the proper path of the dataset
```
cp yolo_training/dataset/data.yaml data/huggingface/
mkdir datasets
mv data/huggingface/* datasets/
```

#### B) Training