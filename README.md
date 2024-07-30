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
Considering only `Car` label here.
1) Collect the labels from text files and view the distribution by label, and distribution of `Car` by difficulty.

```
python3 dataset/labels_distribution.py
```

This will create a `gold_labels.csv` file under dataset

2) Create Train (70%), Test(15%) and Validation(15%) splits 

```
python3 dataset/create_splits.py
```

This will create 3 files `train_split.csv`, `test_split.csv` and `valid_split.csv`
