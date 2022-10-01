import os
import os.path as osp
from re import T


# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# packages from {repo}/cvg-tools
from tkinter import N

import yaml
from algo_sdk.face_sdk.tools.config import AGE
from py_utils.file_system.file_utils import check_path

# packages from {repo}/algo_sdk
from algo_sdk.face_sdk.face_analyzer import FaceAnalyzer
# import algo_sdk.face_sdk.normalization_utils as face_norm

import data_utils
import cv2
import os
from pathlib import Path

from clean_dataset.face_parsing.test import Evaluator, vis_parsing_maps
from clean_dataset.face_parsing.model import BiSeNet
import torch
import numpy as np

dataset_dir='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/val_430'
# dataset_dir='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/special'



img_paths = data_utils.make_im_set(dataset_dir)
result_dict={}


# init face parsing network
face_parsing_checkpoint_path = 'clean_dataset/face_parsing/res/cp/79999_iter.pth'

# net = BiSeNet(n_classes=19)
# net.load_state_dict(torch.load(face_parsing_checkpoint_path,map_location=torch.device('cpu')))

face_parser = Evaluator(face_parsing_checkpoint_path, False)

def write_result(value, dataset_dir,img):
    save_keys = []
    if value['num_person'] > 1:
        save_keys  += ['_multiple']
    if value['wear_hat_prob'] > 0.5:
        save_keys  += ['_wear_hat']
    if value['blur_score'] < 30: # Smaller than 30 is considered blur, Default is 100
        save_keys  += ['_blur']
    if value['num_person'] == 0:
        save_keys  += ['_no_person']
    if value['if_hat_count'] > 1000:
        save_keys  += ['_hat']
    if value['pitch'] > 10:
        save_keys  += ['_pitch']
    if value['yaw'] > 10:
        save_keys  += ['_yaw']
    if value['roll'] > 10:
        save_keys  += ['_roll']
    # elif value['age'] == 0:
        # save_keys  += ['_no_age']

    # save to file
    if len(save_keys) > 0:
        for save_key in save_keys:
            save_dir = dataset_dir+save_key
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            save_path  = os.path.join(save_dir,value['file_name'])
            cv2.imwrite(save_path, img)
    else:
        save_dir = dataset_dir+'_clean'
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        save_path  = os.path.join(save_dir,value['file_name'])
        cv2.imwrite(save_path, img)

def if_hat(parsing,image):
    pixel_count = 0
    if 18 in np.unique(parsing): pixel_count = vis_parsing_maps(image, parsing, stride=1, save_im=False)
    return pixel_count

for img_path in img_paths:
    name = img_path.split('/')[-1]
    result_dict[img_path] = {}

    img = cv2.imread(img_path)
    attr_dict = FaceAnalyzer.detect_faceAttr(img)
    pose = FaceAnalyzer.detect_pose(img)

    # Hat detection
    parsed_info,image_parse = face_parser.parse(img_path,True)
    if_hat_count = if_hat(parsed_info,image_parse)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = data_utils.variance_of_laplacian(gray)

    # gender_type = FaceAnalyzer.detect_gender(img)
    # print("gender_type",gender_type)
    
    num_person = len(attr_dict['expression'])
    wear_hat_prob = [attr_dict['expression'][i]['wear_hat_prob'] for i in range(num_person)]
    age           = [attr_dict['expression'][i]['age'] for i in range(num_person)]
    result_dict[img_path]['file_name'] = name
    result_dict[img_path]['num_person'] = num_person
    result_dict[img_path]['wear_hat_prob'] = max(wear_hat_prob) if num_person > 0 else 0
    result_dict[img_path]['blur_score'] = blur_score
    result_dict[img_path]['age'] = max(age) if num_person > 0 else 0
    result_dict[img_path]['if_hat_count'] = if_hat_count
    result_dict[img_path]['pitch'] = pose[0] if pose !=None else 0
    result_dict[img_path]['yaw']   = pose[1] if pose !=None else 0
    result_dict[img_path]['roll']  = pose[2] if pose !=None else 0


    write_result(result_dict[img_path], dataset_dir, img)

    a=1


        
        


# for key, value in result_dict.items():
#     save_keys = []
#     if value['num_person'] > 1:
#         save_keys  += ['_multiple']
#     elif value['wear_hat_prob'] > 0.5:
#         save_keys  += ['_wear_hat']
#     elif value['blur_score'] > 0.5:
#         save_keys  += ['_blur']
#     elif value['num_person'] == 0:
#         save_keys  += ['_no_person']
#     elif value['age'] == 0:
#         save_keys  += ['_no_age']

#     img = cv2.imread(key)
#     # save to file
#     if len(save_keys) > 0:
#         for save_key in save_keys:
#             save_dir = dataset_dir+save_key
#             Path(save_dir).mkdir(exist_ok=True, parents=True)
#             save_path  = os.path.join(save_dir,value['file_name'])
#             cv2.imwrite(save_path, img)
#     else:
#         save_dir = dataset_dir+'_clean'
#         Path(save_dir).mkdir(exist_ok=True, parents=True)
#         save_path  = os.path.join(save_dir,value['file_name'])
#         cv2.imwrite(save_path, img)
        
