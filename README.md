# Bounding Box generation using Image Segmentation.

This repository provides training and inferencing modules for Pneumonia Detection. This aim of this project is to provide a cookie cutter approach to generate bounding box predictions for Chest X Rays.

There are many object detection techniques are available today for generation of bounding boxes like YoloV5, R-CNN, Faster R-CNN etc. Segmentation is primarily a technique which marks 
Here I am using segmentation to generate bounding boxes. 

## Dataset Used

BIMCV COVID-19+ (https://www.kaggle.com/c/siim-covid19-detection/data).

## Data Set up

The bounding box and label information is provided in csv files. For segmentation, masked images are generated from the given information which is used for model generation.

Following notebooks contain the code for data set up.
- zip-to-dcm.ipynb - Extracts dicom files from zip files.
- dcm-to-png.ipynb - Extracts png files from dicom files.
- png-to-mask.ipynb - Creates mask images for corresponding X Rays using information provided in excel files.

## Usage

## Architecture Used


## For advanced users

Learning rate finder

## Method

