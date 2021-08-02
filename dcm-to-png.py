#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install pydicom')


# In[2]:


import pydicom
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image


# In[3]:


outdir = './data-siim/train/images/'
dcm_dirpath = "./data-siim/train-dcm/"


# In[4]:


train_dcm_files = os.listdir(dcm_dirpath)


# In[5]:


len(train_dcm_files)


# In[6]:


from pydicom import dcmread, read_file
def save_png_from_dicom(dicompath, target_dir):
    target_file_path = target_dir + Path(dicompath).stem + ".png"
    if not(os.path.exists(target_file_path)):
        dicom = read_file(dicompath, stop_before_pixels=False)
        ims = dicom.pixel_array
        #print(target_file_path, ims.shape)
        norm = (ims.astype(np.float)-ims.min())*255.0 / (ims.max()-ims.min())
        Image.fromarray(norm.astype(np.uint8)).save(target_file_path) 


# In[ ]:


train_dcm_files = os.listdir(dcm_dirpath)
i_exp = 0
for f in tqdm(train_dcm_files):   
    try:
        save_png_from_dicom(dcm_dirpath + f, outdir)
    except Exception as e:
        i_exp += 1
print("failed to convert:" ,i_exp , " files")


# In[ ]:


# Remove those images which have some problems.
path = Path(train_path).rglob("*.png")
for img_p in path:
    try:
        img = PIL.Image.open(img_p)
        img.load()
    except IOError:
            print(img_p)
            #os.remove(img_p)

