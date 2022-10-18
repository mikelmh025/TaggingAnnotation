import json
import data_utils
import cv2
import numpy as np
import os

root = '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/'
image_dir = root + 'bitmoji asset_version3'
json_dir = root + 'bitmoji asset_version2_label4'
# mkdir save_dir
save_json_dir = root + 'bitmoji asset_version3_label'
os.makedirs(save_json_dir, exist_ok=True)

for image_path in data_utils.make_im_set(image_dir):
    image_name = image_path.split('/')[-1].split('.')[0]
    json_path = json_dir+'/'+image_name+'.json'
    # save json
    with open(save_json_dir+'/'+image_name+'.json', 'w') as f:
        json.dump(json.load(open(json_path)), f)
