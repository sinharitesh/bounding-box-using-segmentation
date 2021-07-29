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


codes = ['none', 'typical', 'indeterminate', 'atypical']
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['none']


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Training Parser')
    parser.add_argument('--train_dir', type=str , help='training folder,should have images and masks underneath')
    parser.add_argument('--mode', type=str ,default = "partial",  help='mode - partial or full, to train on a subset of images(50)')
    args = parser.parse_args()
    return(args)

def get_subset_data(path):
    fnames = get_image_files(path)
    return fnames[0:50]

def acc_seg(inp, targ):
    if ((inp is None) or (targ is None)): return(0)
    g_inp = inp
    g_targ = targ
    targ = targ.squeeze(1)
    mask = targ != void_code
    ret = (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()
    if math.isnan(ret): ret = 0
    return (ret)


if __name__ == "__main__":
    
    args = parse_args()
    train_dir = args.train_dir
    path = Path(train_dir)
    
    path_im = path/'images'
    path_lbl = path/'masks'
    mode = args.mode
    print( "RES", train_dir)
    
    dblock = None
    dls    = None
    get_msk = lambda o: path/'masks'/f'{o.stem}{o.suffix}'
                
    if (mode == "partial"):
        print("partial mode, pass --mode = full for full training") 
        dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes = codes)),
            get_items=get_subset_data,
            splitter=RandomSplitter(),
            get_y= get_msk,
            item_tfms=Resize(768, method = "squish"),
            batch_tfms= [Normalize.from_stats(*imagenet_stats)]
            )
    if (mode == "full"):
        print("training in full mode.") 
        dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes = codes)),
            get_items=get_image_files,
            splitter=RandomSplitter(),
            get_y= get_msk,
            item_tfms=Resize(768, method = "squish"),
            batch_tfms= [Normalize.from_stats(*imagenet_stats)]
            )
    
    dls = dblock.dataloaders(path/'images', bs=2)
    dls.vocab = codes      
    opt = ranger
    learn = unet_learner(dls, resnet34, metrics=acc_seg, self_attention=True, act_cls=Mish, opt_func=opt, cbs=CSVLogger())
    lr = 1e-3
    learn.fit_flat_cos(1, slice(lr))
    model_path = learn.save("seg-siim-test-img1000-resnet34")
    print("model trained, saved at:", model_path)
                
                
                
                