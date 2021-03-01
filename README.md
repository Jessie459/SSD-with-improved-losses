# CIOU-SSD

## Description

The code is derived from [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) 
and I replace smooth l1 localization loss with complete-iou localization loss.

## dataset

The dataset should be in PASCAL VOC2017 style.

    |--Annotations
        |-- 000001.xml
        |-- 000002.xml
        |-- 000003.xml
        |-- ...
    |--ImageSets
        |-- Main
            |-- train.txt
            |-- trainval.txt
            |-- val.txt
            |-- test.txt
    |-- JPEGImages
        |-- 000001.jpg
        |-- 000002.jpg
        |-- 000003.jpg
        |-- ...

## Train

You can specify the following parameters:

    --resume: optional, a path to the checkpoint
    --device: 'cpu' or 'cuda'
    --data_root: directory that contains Annotations, ImageSets and JPEGImages
    --save_root: directory that contains models
    --image_set: should be consistent with the .txt file name in ImageSets/Main, e.g. 'train'
    --save_freq: epoch frequency of saving
    --print_freq: iteration frequency of printing
    --num_epochs: number of epochs
    --batch_size: batch size
    --lr: initial learning rate
    --alpha: weighting factor used in ciou localization loss

for example:

```python train.py --device='cuda:0' --data_root='data' --save_root='weights' --num_epochs=160 --batch_size=16 --lr=0.001 --alpha=2.0```

## Evaluate

You can specify the following parameters:

    --device: 'cpu' or 'cuda'
    --data_root: directory that contains Annotations, ImageSets and JPEGImages
    --model_root: directory that contains models
    --model_name: model name
    --image_set: should be consistent with the .txt file name in ImageSets/Main, e.g. 'test'
    --batch_size: batch size

for example:

```python evaluate.py --device='cuda:0' --data_root='data'  --model_root='weights' --model_name='ssd300_epochs_160.pth' --batch_size=16```

## Result

I select 400 elephant images from COCO dataset and add them to the PASCAL VOC2017 dataset. 
I trained the model for 160 epochs and the results are as follows:

| class | average precision |
| :----: | :----: |
| aeroplane | 0.7041 |
| bicycle | 0.7868 |
| bird | 0.6811 |
| boat | 0.5871 |
| bottle | 0.3600 |
| bus | 0.7861 |
| car | 0.8256 |
| cat | 0.8365 |
| chair | 0.4731 |
| cow | 0.7648 |
| dinningtable | 0.6815 |
| dog | 0.8279 |
| elephant | 0.8267 |
| horse | 0.8409 |
| motorbike | 0.7948 |
| person | 0.7403 |
| pottedplant | 0.4402 |
| sheep | 0.6889 |
| sofa | 0.7173 |
| train | 0.8191 |
| tvmonitor | 0.7079 |
| **mAP** | **0.7091** |

## Reference

> Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//European conference on computer vision. Springer, Cham, 2016: 21-37.  
> Zheng Z, Wang P, Liu W, et al. Distance-IoU loss: Faster and better learning for bounding box regression[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(07): 12993-13000.  

