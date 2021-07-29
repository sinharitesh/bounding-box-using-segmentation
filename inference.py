import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import patches, patheffects
from fastai.vision.all import *
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
import random
import fastai
import matplotlib.pyplot as plt
from tqdm import tqdm
from common_fns import *


test_path  =  './test/'


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--test_dir', type=str , default = "./test/" , help='directory containing the images on which inference is required.')
    parser.add_argument('--mode', type=str ,default = "generate-images",  help='mode - created images with bounding boxes drawn over')
    args = parser.parse_args()
    return(args)


if __name__ == "__main__":
    
    args = parse_args()
    test_path  =  args.test_dir + "images/"  #  './test/'
    if not os.path.exists("./output/"):
        os.makedirs("./output")

    test_images = os.listdir(test_path)
    for i, img in enumerate(test_images):
        imgpath = test_path + img
        save_original_image(imgpath)
        save_predicted_image(imgpath)
    print("Inference completed for:", i, "images. Source:", test_path) 
    
                
                
                
                