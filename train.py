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
from datetime import datetime


codes = ['none', 'typical', 'indeterminate', 'atypical']
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['none']
model_prefix = "siim-segmentation-resnet34"


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Training Parser')
    parser.add_argument('--train_dir', type=str , help='training folder,should have images and masks underneath')
    parser.add_argument('--mode', type=str ,default = "partial",  help='mode - partial or full, to train on a subset of images(50)')
    parser.add_argument('--use_model', type=str ,default = "",  help='model for transfer learning')
    parser.add_argument('--learning_rate', type=float ,default = .003,  help='learning rate')
    parser.add_argument('--epochs', type=int ,default = 1,  help='learning rate')
    
    args = parser.parse_args()
    return(args)

def get_subset_data(path):
    fnames = get_image_files(path)
    return fnames[0:20]

def acc_seg(inp, targ):
    if ((inp is None) and (targ is None)): return(1)
    if ((inp is None) or (targ is None)): return(0)
    targ = targ.squeeze(1)
    mask = targ != void_code
    ret = (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()
    if math.isnan(ret): ret = 0
    return (ret)

def update_training_csv(learning_rate, model_name):
    try:
        df_curr = pd.read_csv('./history.csv')
        df_curr['learning_rate'] = learning_rate
        df_curr['model_name'] = model_name

        if os.path.exists("./train_history.csv"):
            df_th =pd.read_csv("./train_history.csv")
            df_th.append(df_curr).to_csv("train_history.csv", index = False)
        else:
            df_curr.to_csv("train_history.csv", index = False)
    except:
        pass


if __name__ == "__main__":
    
    args = parse_args()
    train_dir = args.train_dir
    path = Path(train_dir)
    
    path_im = path/'images'
    path_lbl = path/'masks'
    mode = args.mode
    lr = args.learning_rate  #1e-3
    epochs = args.epochs
    
    use_model = args.use_model
    if (use_model != ""):
        model_file = "./models/" + use_model
        if not os.path.exists(model_file):
            print("model", model_file, "is not available, quitting!")
        else:    
            use_model = use_model.replace(".pth", "")
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
            if (use_model != ""):
                learn = learn.load(use_model)
                print(use_model, "loaded.")
            
            
            learn.fit_flat_cos(epochs, slice(lr))
            fname = datetime.now().strftime("%d%m%Y-%H%M%S")
            model_name = model_prefix + "-" + fname
            model_path = learn.save(model_name)
            print(f"model trained with lr:{lr}, epochs:{epochs}, saved at:{model_path}")
            update_training_csv(lr, model_name)
                
                
                
