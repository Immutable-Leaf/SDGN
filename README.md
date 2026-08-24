Prepare Datasets

    (1) RegDB Dataset: The RegDB dataset can be downloaded from "https://www.kaggle.com/datasets/xqq027/reg-db"

    (2) SYSU-MM01 Dataset: The SYSU-MM01 dataset can be downloaded from "https://www.kaggle.com/datasets/coconutjean/sysumm01/data".
        run "python pre_process_sysu.py" to pepare the dataset, the training data will be stored in ".npy" format.

Training

    python train.py --gpu 'your device id' --dataset 'sysu or regdb' 

Example

    python train.py --gpu 0 --dataset sysu

Evaluation

    python test.py --mode 'mode for SYSU-MM01' --resume 'model_path' --gpu 'your device id' --dataset 'sysu or regdb'
