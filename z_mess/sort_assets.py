import data_utils
import cv2
import numpy as np
import json


ass_img_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/images'
asset_label_path = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/820_faceattribute_round2_asset_translate_soft.json'
column = 5

ass_img_paths = data_utils.make_im_set(ass_img_dir)

# Read json
with open(asset_label_path) as f:
    asset_labels = json.load(f)


def dict2int(data_dict):
    total_val,counter = 0,0
    for key in data_dict:
        value = key.split('-')[0]
        total_val += int(value)*data_dict[key]
        counter += data_dict[key]
    return total_val/counter

asset_data_dict = {}
for asset_path in ass_img_paths:
    name = str(asset_path.split('/')[-1])
    label = asset_labels[name]

    top_length = dict2int(label['top_length'])
    side_length = dict2int(label['side_length'])
    length = top_length*0.3 + side_length*0.7
    # length = label['top_length']+label['side_length']

    asset_data_dict[asset_path] = [length,top_length,side_length]
    # print(asset_labels[id])

# sort asset_data_dict by value
asset_data_dict = {k: v for k, v in sorted(asset_data_dict.items(), key=lambda item: item[1])}

# Read images
asset_imgs = []
asset_img_ids = []
for key in asset_data_dict:
    img_id = key.split('/')[-1].split('.')[0]
    asset_imgs.append(data_utils.read_img(key,height=512))
    asset_img_ids.append(img_id)

print(asset_img_ids)

# Create image rows
image_rows = []
for i in range(len(asset_imgs)//column+1):
    image_rows.append(data_utils.horizontal_cat(asset_imgs[i*column:(i+1)*column],column))

# Stack created rows
asset_concat = data_utils.vertical_cat(image_rows)

cv2.imwrite('concat_img_sorted.jpg', asset_concat)