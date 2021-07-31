# Bounding Box generation using Image Segmentation.

This repository provides training and inferencing modules for Pneumonia Detection. This aim of this project is to provide a cookie cutter approach to generate bounding box predictions for Chest X Rays.

There are many object detection techniques are available today for generation of bounding boxes like YoloV5, R-CNN, Faster R-CNN etc. 
Segmentation is primarily a technique which marks. Here I am using segmentation to generate bounding boxes.

This project provides two functions:

1. Inferencing on Chest XRay Images to find Bounding box and associated labels. (A pretrained model is provided, but users are encouraged to train their own models based on the training script provided as below.
2.  Training of model : The data setup is explained and code is made available in the Dataset up section below. The data for traning is required in two folders names images and masks under a parent folder, say train. At time of training, the location of this folder (train) needs to be passed to the training script (train.py).

## Dataset Used

BIMCV COVID-19+ (https://www.kaggle.com/c/siim-covid19-detection/data).

## Data Set up

The bounding box and label information is provided in csv files. For segmentation, masked images are generated from the given information which is used for model generation.

Following notebooks contain the code for data set up.
- zip-to-dcm.ipynb -  Extracts dicom files from zip files.
- dcm-to-png.ipynb -  Extracts png files from dicom files. Code for cleaning the corrupted images is also available.
- png-to-mask.ipynb - Creates mask images for corresponding X Rays using information provided in excel files.

## Usage

### Training 
Get the repository
  
  git clone https://github.com/sinharitesh/bounding-box-by-segmentation.git
  cd bounding-box-by-segmentation   (Change the directory)
  Create a directory "models" in current folder - mkdir models 
  Change directory to models

  Download a pre trained model provided for this purpose using following in the models directory :
  
  gdown https://drive.google.com/uc?id=1-9uL0NaYMWowQdFFVzHS6Jfv3OB_ZhQp.
  
  This will download a model named "siim-seg-011-resnet34-colab.pth".
  
  Comeback to root directory (one directory up)
  Run the following command:
  
  python train.py --train_dir ../data-siim/train/ --use_model siim-seg-011-resnet34-colab.pth --epochs 1 --learning_rate .01
  
  Training will start, after the training is finished, a model named siim-segmentation-resnet34-<DDMMYYYY.-<HHMMSS>.pth will be available in models directory.
  

## Architecture Used


## For advanced users

Learning rate finder

## Method

