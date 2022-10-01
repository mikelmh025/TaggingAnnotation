import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from os.path import exists

IMAGE_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG','png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)

# Only one level of file
def make_im_set(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
        break
    return images


JSON_EXTENSIONS = [
    '.json', '.JSON'
]

def is_json_file(filename):
    return any(filename.endswith(extension) for extension in JSON_EXTENSIONS)

# Only one level of file
def make_json_set(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_json_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
        break
    return images

# def get_int_from_str(str_list):
#     out =[ ]
#     for str in str_list:
#         digit_list = [c for c in str if c.isdigit()]
#         digit = ""
#         for item in digit_list:
#             digit+=item
#         out.append(int(digit))

#     return out

# # Input: [Dim, Batch]
# def get_entropy(x,bin_range=3):
#     bin,edge = np.histogram(x[:], bins=range(bin_range))
#     bin = bin+0.0000001
#     p = bin/bin.sum(axis=0, keepdims=True)
#     entro = (-p*np.log2(p)).sum(axis=0)
#     entro = round(entro,4)
#     return entro

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
 


# data: list of int
def get_entropy(data,category_count):
    assert len(data) > 0, "data is empty. Or input is not list"
    bin, edge = np.histogram(data, bins=category_count)
    bin = bin + 1e-10
    p   = bin / np.sum(bin,keepdims=True)
    entropy = -np.sum(p * np.log2(p),axis=0)
    return entropy

ce_loss = nn.CrossEntropyLoss()
def get_cross_entropy(data_list,target,category_count):
    assert len(data_list) > 0, "data is empty. Or input is not list"
    entropy = []
    for data in data_list:
        y = torch.LongTensor([[data]])
        one_hot = torch.FloatTensor(1, category_count).zero_()
        one_hot.scatter_(1, y, 1)

        entropy += [ce_loss(one_hot, torch.tensor([target]).long()).item()]
    return sum(entropy)/len(entropy)

def get_accuracy(data_list,target):
    assert len(data_list) > 0, "data is empty. Or input is not list"
    accuracy = []
    for data in data_list:
        y = torch.LongTensor([[data]])
        accuracy += [torch.sum(y == torch.tensor([target]).long()).item()]
    return sum(accuracy)/len(accuracy)

def get_mse(data_list,target):
    assert len(data_list) > 0, "data is empty. Or input is not list"
    mse = []
    for data in data_list:
        mse += [torch.sum((torch.tensor([data]) - torch.tensor([target]))**2).item()]
    return sum(mse)/len(mse)

def attr2int(attr):
    return int(attr.split('-')[0])

def intersection_list(list1,list2):
    return list(set(list1) & set(list2))

def concat_list_image(matched_asset_paths,matched_titles=None):
    assert len(matched_asset_paths) > 0, "No matched asset found"
    #use cv2
    imgs = [cv2.imread(str(asset_path)) for idx, asset_path in enumerate(matched_asset_paths)]
    
    try:
        height_list = [img.shape[0] for img in imgs]
    except:
        # throw error if image is not read
        raise Exception("Image not read")
    max_height = max(height_list)
    width_list = [img.shape[1] for img in imgs]
    max_width = max(width_list)
    resized_imgs = []
    
    for idx, img in enumerate(imgs):
        ratio = max_height/img.shape[0]
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        img = cv2.putText(img, matched_titles[idx], (20,img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 8)
        if idx == 0:
            im_concat = img
        else:
            im_concat = np.concatenate((im_concat, img), axis=1)
    return im_concat

def fix_image_subscript(img_path):
    sub_scripts = ['.jpg','.JPG','.png','.PNG','.jpeg','.JPEG']
    if exists(img_path):
        return img_path
    else:
        subscirpt = '.'+img_path.split('.')[-1]
        path_name = img_path.replace(subscirpt,'')

        for option in sub_scripts:
            if exists(path_name+option):
                return path_name+option

        raise Exception("Image not read")


def horizontal_cat(imgs,column):
    if len(imgs) <column:
        imgs += [np.zeros_like(imgs[0])]*(column-len(imgs))
    for i in range(len(imgs)):
        concat_img = cv2.hconcat([concat_img, imgs[i]]) if i != 0 else imgs[i]
    return concat_img

def vertical_cat(imgs):
    min_width = max([img.shape[1] for img in imgs])

    for i in range(len(imgs)):
        ratio = min_width/imgs[i].shape[1]
        img_ = cv2.resize(imgs[i], None, fx=ratio, fy=ratio)

        concat_img = cv2.vconcat([concat_img, img_]) if i != 0 else img_
    return concat_img

def read_img(path,height=512):
    img = cv2.imread(path)
    ratio = height/img.shape[0]
    img = cv2.resize(img, None, fx=ratio, fy=ratio)         # reshpae img to 512 height
    return img