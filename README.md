# Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing

## Pipeline

![framework](/figs/1.jpg)


## Installation
1. Clone the repository.
    ```bash
    https://github.com/xxx
    ```

2. Install PyTorch 1.12.0 and torchvision 0.13.0.
    ```bash
    conda install -c pytorch pytorch torchvision
    ```

3. Install the other dependencies.
    ```bash
    pip install -r requirements.txt
    ```
    
## Prepare

The final file path should be the same as the following (please check it carefully):
```
┬─ save_models
│   ├─ indoor
│   │   ├─ DIACMPN-dehighlight-Indoor.pth
│   │   ├─ DIACMPN-depth-Indoor.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ RD
    │   ├─ train
    │   │   ├─ input
    │   │   │   └─ ... (image filename)
    │   │   │─ mask
    │   │   │   └─ ... (image filename)
    │   │   └─ target
    │   │       └─ ... (image filename)
    │   └─ test
    │   │   ├─ input
    │   │   │   └─ ... (image filename)
    │   │   └─ target
    │   │       └─ ... (image filename)
    └─ ... (dataset name)
```

## Training

To customize the training settings for each experiment, navigate to the `configs` folder. Modify the configurations as needed.

After adjusting the settings, use the following script to initiate the training of the model:

```sh
CUDA_VISIBLE_DEVICES=X python train.py --model (model name) --model_detection (detection_model name) --dataset (dataset name) --exp (exp name)
```

For example, we train the DIACMPN-dehighlight-Indoor on the RD:

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --model Highlight-Removal --model_detection Highlight-Removal-detection --dataset RD --exp Highlight-Removal
```

## Evaluation

Run the following script to evaluate the trained model with a single GPU.


```sh
CUDA_VISIBLE_DEVICES=X python test.py --model (model name) --model_detection (detection_model name) --dataset (dataset name) --exp (exp name)
```

For example, we test the DIACMPN-dehighlight-Indoor on the SOTS indoor set:

```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model Highlight-Removal --model_detection Highlight-Removal-detection --dataset RD --exp Highlight-Removal
```


# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Zhou Shen
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: zhoushennn@163.com
