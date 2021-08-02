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

def acc_seg(inp, targ):
    if ((inp is None) or (targ is None)): return(0)
    g_inp = inp
    g_targ = targ
    targ = targ.squeeze(1)
    mask = targ != void_code
    ret = (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()
    if math.isnan(ret): ret = 0
    return (ret)
# Following needs to come from config file.

valid_path =  './valid/'
test_path  =  './test/'
default_model_name = "siim-seg-011-resnet34-colab"

codes = ['none', 'typical', 'indeterminate', 'atypical']
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['none']

lst_categories = []
lst_categories.append({'id': 0, 'name' : 'none'})
lst_categories.append({'id': 1, 'name' : 'typical'})
lst_categories.append({'id': 2, 'name' : 'indeterminate'})
lst_categories.append({'id': 3, 'name' : 'atypical'})


path_im  =  Path(valid_path)/'images'
get_msk  =   lambda o: Path(valid_path)/'masks'/f'{o.stem}{o.suffix}'

siimblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes = codes)),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y= get_msk,
    item_tfms=Resize(768, method = "squish"),
    batch_tfms= [Normalize.from_stats(*imagenet_stats)]
    )

dls = siimblock.dataloaders(path_im, bs=2)

opt = ranger
learn = unet_learner(dls, resnet34, metrics = acc_seg, self_attention=True, act_cls=Mish, opt_func=opt)
learn = learn.load(default_model_name)
print("loaded:", default_model_name)


def deleted_get_bboxes(str_label):
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

def get_xyxy_fromxywh(bb):
    w = bb[2] 
    h = bb[3] 
    x = bb[0]
    y = bb[1]
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    return([x1, y1, x2, y2])

def get_xywh_fromxyxy(bb):
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    x = bb[0] + w * .5
    y = bb[1] + h *.5
    return([x, y, w, h])

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], 
                         fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white',
                   fontsize=sz, weight='bold')
    draw_outline(text, 1)
    
def bb_hw2(a): return np.array([a[0],a[1],a[2]-a[0], a[3]-a[1]]) # this is good.

def get_bbarea(prbbs):
    bbarea = []
    for bb in prbbs:
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]
        ar = width*height
        bbarea.append(ar)
    return(bbarea)

def get_predicted_bboxes(masked_image):
    lbl_0 = label(masked_image) 
    props = regionprops(lbl_0)
    propbbs = []
    labels = []
    for prop in props:
            tmp = []
            tmp.append(prop.bbox[1])
            tmp.append(prop.bbox[0])
            tmp.append(prop.bbox[3])
            tmp.append(prop.bbox[2])
            #print(, prop.bbox[0], prob.bbox[3], prob.bbox[2])
            propbbs.append(tmp)
            labels.append(prop.label)
            #print('Found bbox', prop.bbox, prop.label) 
    return(propbbs, labels)
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

def plot_image_bb(im, ax, bbs, lbl):
    ax = show_image(im = im, ax = ax, cmap=plt.get_cmap('gray'))
    for i, bx in enumerate(bbs):
        b = bb_hw2(bx)
        draw_rect(ax, b)
        draw_text(ax, b[:2], lbl)
        
        
def get_category_label( _id = 0, categories = lst_categories):
    ele = next(item for item in categories if item['id'] == _id)
    return(ele['name'])

def get_cat_name(dcmfilename, categories = lst_categories):
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
    return(label)


def get_cat_id(png_file, categories = lst_categories):
    label = read_label(png_file)
    cat_id = get_category_by_name(categories, label)
    return(cat_id)

def get_masked_image(fname):
    #fname = few_images[0]
    im = Image.open(fname)
    masked_image = np.zeros(im.shape)
    png_file = fname.stem
    bbstr = (read_bbstr(png_file))
    bbs = get_bboxes(bbstr)
    cat_id = get_cat_id(png_file)
    for bb in bbs:
        bb = list(map(round, (bb))); #print(bb)
        masked_image[bb[1]: bb[3], bb[0]: bb[2]] =  cat_id
    return(masked_image)

def plot_image_bb(im, ax, bbs, lbl):
    ax = show_image(im = im, ax = ax, cmap=plt.get_cmap('gray'))
    for i, bx in enumerate(bbs):
        b = bb_hw2(bx)
        draw_rect(ax, b)
        draw_text(ax, b[:2], lbl)
        
        
def show_new_2(im, filename, np_mask, label, prob, bb_resized):
    caption = label  + " " +  str(prob)
    path_im = test_path + "images/"
    fl = f"{path_im}/{filename}"
    _ , (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize= (12,12))
    bbstr = read_bbstr(Path(fl).stem)
    bbs = get_bboxes(bbstr)
    plot_image_bb(im = im, ax = ax1, bbs = bbs, lbl = "o")
    masked_image = get_masked_image(Path(fl))
    show_image(im = masked_image, ax = ax2);
    show_image(im = np_mask, ax = ax3);
    plot_image_bb(im = im, ax = ax4, bbs = bb_resized, lbl = caption)
    
    
def get_category_by_name(categories = lst_categories , name = "none"):
    ele = next(item for item in categories if item['name'] == name)
    return(ele['id'])


def get_prob_and_label(pred_arx, pred_one):
    predicted_class_label = ""
    predicted_class_prob  = 0
    output_cls = len(pred_arx.unique())
    if (output_cls > 1) :
        pr_1 = pred_one[1, pred_arx == 1].mean()
        pr_2 = pred_one[2, pred_arx == 2].mean()
        pr_3 = pred_one[3, pred_arx == 3].mean()

        if (np.isnan(pr_1.numpy())):
            pr_1 = tensor(0)
        if (np.isnan(pr_2.numpy())):
            pr_2 = tensor(0)
        if (np.isnan(pr_3.numpy())):
            pr_3 = tensor(0)
        max_arg = tensor([pr_1, pr_2, pr_3]).argmax(dim=0)
        max_arg = max_arg.numpy()
        predicted_class_index = max_arg + 1
        predicted_class_label = get_category_label(predicted_class_index)
        predicted_class_prob = max(pr_1, pr_2, pr_3)
        predicted_class_prob = predicted_class_prob.numpy().round(1)
    return(predicted_class_prob , predicted_class_label )

def get_resized_bb(prbbs, orig_size, curr_size):
    bb_return = []
    bb_area_return = []
    x_ = orig_size[0]
    y_ = orig_size[1]
    x_scale = x_/curr_size[0]#/x_
    y_scale = y_/curr_size[1]#/y_
    for bb_temp in prbbs:
        bb_resized = [bb_temp[0] * x_scale, bb_temp[1] * y_scale,  bb_temp[2] * x_scale, bb_temp[3] * y_scale]
        bb_resized = [int(x) for x in bb_resized]
        width = bb_temp[2] - bb_temp[0]
        height = bb_temp[3] - bb_temp[1]
        bb_ar = width*height
        bb_return.append(bb_resized)
        bb_area_return.append(bb_ar)
    return(bb_return, bb_area_return)

def get_bb_adjusted(bb, width_to=140, height_to = 140):
    bb_temp = bb
    bb_adjusted = bb
    #bb_temp = [136, 404, 194, 472]
    #width_to = 140
    #height_to = 140
    width = bb_temp[2] - bb_temp[0]
    height = bb_temp[3] - bb_temp[1]
    if (width*height) < (width_to*height_to):
        w_diff = width_to - width
        h_diff = height_to - height
        bb_adjusted = [  bb_temp[0] - w_diff *.5
                       , bb_temp[1] - h_diff *.5
                       , bb_temp[2] + w_diff *.5
                       , bb_temp[3] + h_diff *.5 ]
        bb_adjusted = [int(b) for b in bb_adjusted]
#     print(bb_temp, bb_adjusted)
    return(bb_adjusted)

def get_filtered_and_resized_bbs(im , np_mask):
    prbbs, predlabels = get_predicted_bboxes(np_mask)
    
    areas = get_bbarea(prbbs)
    #print("1.", prbbs, predlabels, areas)
    filtered_bbs = filter_bboxes(areas, prbbs, thresh = 200)
    bb_resized , bb_area = get_resized_bb(filtered_bbs, im.shape, np_mask.shape)
    return(bb_resized, bb_area, prbbs)

def filter_bboxes(bbareas, prbbs, thresh= 20000):
# The threshold is set for 768 size while training , this needs to be adjusted when image size changes.
# Also if there are only 1 small bounding box , then should inlcude this with some minimum area, that will
# come from trial and error. But later TODO.
    ret_bbs = []
    #if (len(bbareas) > 3): # This will also change, take the top two.
    for i, area in enumerate(bbareas):
        if (area > thresh):
            ret_bbs.append(prbbs[i])
    
    # square root of 20000 = 140
    if (len(bbareas) == 1):
        for i, area in enumerate(bbareas):
            if (area < thresh):
                adj_prbb = get_bb_adjusted(prbbs[0]) # There is only one area here.
                ret_bbs.append(adj_prbb)
                
    # need to deal with two boxes when all are left out, there could be many possibilities here. Later on.
    #print("filter_bboxes", bbareas, prbbs, ret_bbs)
    return(ret_bbs) 

def read_bbstr(f ):
    ret = ""
    label_folder   = test_path + 'labels/'
    try:
        with open(label_folder + f.split('.')[0] + "-boundingbox.txt", 'r') as fl:
            ret = (fl.readline())
    except:
        pass
    return(ret)

def read_label(f ):
    ret = ""
#     label_folder   = test_path/'labels/'
    label_folder   = test_path + 'labels/'
    try:
        with open(label_folder + f.split('.')[0] + "-label.txt", 'r') as fl:
            ret = (fl.readline())
    except:
        pass
    return(ret)

# def save_image_bb(im, ax, bbs, lbl, fsave = "default.png"):
#     ax = show_image(im = im, ax = ax, cmap=plt.get_cmap('gray'))
#     for i, bx in enumerate(bbs):
#         b = bb_hw2(bx)
#         draw_rect(ax, b)
#         draw_text(ax, b[:2], lbl)
     
#     plt.savefig(fsave)
#     plt.close()

def save_original_image(str_fpath):
    dirsave =  "./output/"
    im =  Image.open(str_fpath)
    fname = Path(str_fpath).stem
    bbs = [[0, 0, 0 , 0]]
    label = ""
    try:
        bbstr = read_bbstr(fname)
        label  = read_label(fname)
        bbs = get_bboxes(bbstr)
    except:
        pass
    fsave = f"{dirsave}{fname}_orig.png"
    save_image_bb(im = im, ax = None, bbs = bbs, lbl = label, fsave = fsave)

def save_predicted_image(str_fpath):
    dirsave =  "./output/"
    im =  Image.open(str_fpath)
    fname = Path(str_fpath).stem
    dl = learn.dls.test_dl(str_fpath)
    preds = learn.get_preds(dl=dl) # forward propagation
    pred_1 = preds[0][0]
    pred_arx = pred_1.argmax(dim=0)
    predicted_class_prob, predicted_class_label = get_prob_and_label(pred_arx, pred_1)
    np_mask = pred_arx.numpy()
    bb_resized, bb_area , prbbs = get_filtered_and_resized_bbs(im, np_mask)
    fsave = f"{dirsave}{fname}_pred.png"
    save_image_bb(im = im, ax = None, bbs = bb_resized, lbl = predicted_class_label, fsave = fsave)
    
def debug_show_one_image(fname = 'sample001.png'):
    label_dir = test_path + 'labels/'
    path_im =   Path(test_path + 'images/')
    fpath =  f"{path_im}/{fname}"
    im =  Image.open(fpath)
    dl = learn.dls.test_dl(fpath)
    preds = learn.get_preds(dl=dl) # forward propagation
    pred_1 = preds[0][0]
    pred_arx = pred_1.argmax(dim=0)
    predicted_class_prob, predicted_class_label = get_prob_and_label(pred_arx, pred_1)
    np_mask = pred_arx.numpy()
    bb_resized, bb_area , prbbs = get_filtered_and_resized_bbs(im, np_mask)
    print(fname, predicted_class_prob, predicted_class_label, bb_resized, bb_area, prbbs)
    show_new_2(im, fname, np_mask, predicted_class_label, predicted_class_prob, bb_resized)
    
    
def save_image_bb(im, ax, bbs, lbl, fsave = "default.png"):
    ax = show_image(im = im, ax = ax, cmap=plt.get_cmap('gray'))
    for i, bx in enumerate(bbs):
        b = bb_hw2(bx)
        draw_rect(ax, b)
        draw_text(ax, b[:2], lbl)
     
    plt.savefig(fsave)
    plt.close()