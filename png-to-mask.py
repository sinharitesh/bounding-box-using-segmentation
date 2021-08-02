#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install fastai==2.3.1 > /dev/null')


# In[2]:


import fastai; print(fastai.__version__)


# In[3]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import os


# In[ ]:


from fastai.vision.all import *
from fastai.medical.imaging import *


# In[6]:


FULL_RUN = False
input_folder = './data-siim/train/'
train_path = './data-siim/train/images/'
mask_path = './data-siim/train/masks/'


# In[7]:


df_image_level = pd.read_csv(f"{input_folder}train_image_level.csv")
df_study_level = pd.read_csv(f'{input_folder}train_study_level.csv')
df_study_level.columns = ["id", "negative", "typical", "indeterminate", "atypical"]
df_study_level['match_id'] = list(map(lambda x: x.replace("_study", ""), df_study_level['id']))
df_image_level['match_id'] = df_image_level['StudyInstanceUID']
df_merge = pd.merge(df_study_level, df_image_level , on = ["match_id"])
df_merge['dcm'] = list(map(lambda x: x.replace("_image", ""), df_merge['id_y']))
df_merge['png'] = list(map(lambda x: x.replace("_image", ""), df_merge['id_y']))


# In[8]:


df_merge.head()


# In[9]:


lst_categories = []
lst_categories.append({'id': 0, 'name' : 'none'})
lst_categories.append({'id': 1, 'name' : 'typical'})
lst_categories.append({'id': 2, 'name' : 'indeterminate'})
lst_categories.append({'id': 3, 'name' : 'atypical'})
lst_categories


# In[10]:


def get_cat_id(dcmfilename, categories = lst_categories):
    df_ret = df_merge[df_merge['dcm']== dcmfilename]
    #df_ret['negative'].values
    label = "none"
    if (df_ret.shape[0] == 1):
        if (df_ret['negative'].values == 1):
            label = "none"
        if (df_ret['indeterminate'].values == 1):
            label = "indeterminate"
        if (df_ret['atypical'].values == 1):
            label = "atypical"
        if (df_ret['typical'].values == 1):
            label = "typical"
    cat_id = get_category_by_name(categories, label)
    return(cat_id)


# In[11]:


def get_category_by_name(categories = lst_categories , name = "none"):
    ele = next(item for item in categories if item['name'] == name)
    return(ele['id'])


# In[12]:


def get_masked_image(fname):
    #fname = few_images[0]
    im = Image.open(fname)
    masked_image = np.zeros(im.shape)
    png_file = fname.stem
    bbstr = (get_bbstr(png_file))
    bbs = get_bboxes(bbstr)
    cat_id = get_cat_id(png_file)
    for bb in bbs:
        bb = list(map(round, (bb))); #print(bb)
        masked_image[bb[1]: bb[3], bb[0]: bb[2]] =  cat_id
    return(masked_image)
    #show_image(masked_image);


# In[13]:


def get_bboxes(str_label):
    l_master = [0,0,1,1]
    try:
        elements = str_label.split(" "); num_labels = len(elements)/6; #print(int(num_labels))
        l_master = []
        for i in range(int(num_labels)):
            l = []
            l.append(elements[i * 6 + 2])
            l.append(elements[i * 6 + 3])
            l.append(elements[i * 6 + 4])
            l.append(elements[i * 6 + 5])
            l = [ float(x) for x in l ]
            l_master.append(l)
    except Exception as e:
            pass
    return(l_master)


# In[14]:


def get_bbstr(pngfilename):
    df_ret = df_merge[df_merge['png'] == pngfilename].reset_index()
    bbstr = ""
    if (df_ret.shape[0] == 1):
        bbstr = df_ret['label'][0]
    return(bbstr)


# In[ ]:


pngfiles = get_image_files(train_path)
for fl in tqdm(pngfiles):
    target_file = mask_path + fl.stem + ".png"
    if not(os.path.exists(target_file)):
        mskfl = get_masked_image(fl)
        Image.fromarray((mskfl).astype(np.uint8)).save(target_file)

