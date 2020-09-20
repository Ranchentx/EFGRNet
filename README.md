# Enriched Feature Guided Refinement Network for Object Detection

By Jing Nie1†, Rao Muhammad Anwer†, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang1‡, Ling Shao  \
† denotes equal contribution，‡ Corresponding author


### Introduction
We propose a single-stage detection framework that
jointly tackles the problem of multi-scale object detection and class imbalance.
Rather than designing deeper networks, we introduce a simple yet effective feature enrichment scheme to produce multi-scale contextual features.
We further introduce a cascaded refinement scheme which first instills multi-scale contextual features into the prediction layers of the single-stage detector
in order to enrich their discriminative power for multi-scale detection. Second, the cascaded refinement scheme counters the class im- balance problem by refining the
anchors and enriched features to improve classification and regression.

## Installation
- Clone this repository. This repository is mainly based mainly based on [SSD_Pytorch](https://github.com/yqyao/SSD_Pytorch.git)

```Shell
    EFGR_ROOT=/path/to/clone/EFGR
    git clone https://github.com/Ranchentx/EFGRNet.git $EFGR_ROOT
```

- The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v0.4.1.
NVIDIA GPUs are needed for testing. After install Anaconda, create a new conda environment, activate the environment and install pytorch0.4.1.

```Shell
    conda create -n EFGRNet python=3.6
    source activate EFGRNet
    conda install pytorch=0.4.1 torchvision -c pytorch
```


- Install opencv and COCOAPI.
```Shell
    pip install opencv-python
    pip install pycocotools
```

- Compile NMS:

```Shell
    cd $EFGR_ROOT/
    ./make.sh
```

- Compile DCN:

```Shell
    ./compile.sh
```


## Download
To evaluate the performance reported in the paper, Pascal VOC and COCO dataset as well as our trained models need to be downloaded.


### VOC Dataset
- Directly download the images and annotations from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/) and put them into $LFIP_ROOT/data/VOCdevkit/.
- Create the `VOCdevkit` folder and make the data(or create symlinks) folder like:

  ~~~
  ${$EFGR_ROOT}
  |-- data
  `-- |-- VOCdevkit
      `-- |-- VOC2007
          |   |-- annotations
          |   |-- ImageSets
          |   |-- JPEGImages
          |-- VOC2012
          |   |-- annotations
          |   |-- ImageSets
          |   |-- JPEGImages
          |-- results
  ~~~

### COCO Dataset
- Download the images and annotation files from coco website [coco website](http://cocodataset.org/#download).
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${$EFGR_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2014.json
          |   |-- instances_val2014.json
          |   |-- image_info_test-dev2015.json
          `-- images
          |   |-- train2014
          |   |-- val2014
          |   |-- test2015
          `-- cache
  ~~~

## Training



```Shell
python train_coco.py --cfg ./configs/EFGRNet_vgg_coco_dcn.yaml
```


## Testing

- Note:
  All testing configs are in EFGRNet_vgg_coco_dcn.yaml, you can change it by yourself.

- To evaluate a trained network:

```Shell
python eval_dcn.py --cfg ./configs/EFGRNet_vgg_coco_dcn.yaml --weights ./eval_weights
```

## Models

* COCO [EFGRNet_VGG320](https://drive.google.com/open?id=1-_x9e4kX3ZJBKzfTKloslJxK2qO8bfkO); [BaiduYun Driver](https://pan.baidu.com/s/1ZPiibo-PnoTJl5HjAl63Pg&shfl=sharepset)
* COCO [EFGRNet_VGG512](https://drive.google.com/open?id=1OVRiYRAyJiErUYsOXPaE12XEXtAV4ZrD); [BaiduYun Driver](https://pan.baidu.com/s/1YvXhhIXdziDV9q3wj9mLRg&shfl=sharepset)


## Citation
Please cite our paper in your publications if it helps your research:

    @article{Jing2019EFGR,
        title = {Enriched Feature Guided Refinement Network for Object Detection},
        author = {Jing Nie, Rao Muhammad Anwer, Hisham Cholakkal, Fahad Shahbaz Khan， Yanwei Pang, Ling Shao},
        booktitle = {ICCV},
        year = {2019}
    }