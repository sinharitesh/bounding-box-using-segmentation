#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ! mkdir ./data-siim


# In[2]:


import os
import cv2
import zipfile
from tqdm import tqdm


# In[34]:


os.chdir('./data-siim')


# In[7]:


#!kaggle competitions download -c siim-covid19-detection
# Use the Kaggle API to download the dataset.
# https://github.com/Kaggle/kaggle-api


# In[35]:


with zipfile.ZipFile('siim-covid19-detection.zip','r') as inzipfile:
    lst_files = inzipfile.namelist()


# In[36]:


lst_files_to_extract =[]
for file in lst_files:
    if (Path(file).suffix == ".dcm"):
        splits = file.split("/")
        if (len(splits) == 4):
            if splits[0] == "train": #Only storing train files
                lst_files_to_extract.append(file)


# In[37]:


len(lst_files_to_extract)


# In[35]:


if not os.path.exists("train-dcm"):
    os.makedirs("train-dcm")


# In[41]:


# dirpath = "./train-dcm/"
# dcm_files = os.listdir(dirpath)
# dcm_files[0]


# In[47]:


# fl = lst_files_to_extract[0]; fl
# #fl = dcm_files[0]
# pathfl = "./train-dcm/" + Path(fl).stem + ".dcm"
# print(pathfl)
# os.path.exists(pathfl)


# In[ ]:


dirpath = "./train-dcm/"
i = 0
with zipfile.ZipFile('siim-covid19-detection.zip', "r") as archive:
    for new in tqdm(lst_files_to_extract):
        new_filename = os.path.join(dirpath, Path(new).name)
        if not(os.path.exists(new_filename)):
            content = archive.open(new).read()
            with open(new_filename, "wb") as outfile:
                outfile.write(content)
                i = i + 1
    print("processed:", i, "files.")

