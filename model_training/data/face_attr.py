import imp
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import data_utils
from os import listdir
from os.path import *
import PIL.Image as Image
from pathlib import Path

import json
# train_face_attr_transform = transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.4801, 0.4250, 0.4140], std=[0.3512, 0.3312, 0.3284])
#     ])

# test_face_attr_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.4801, 0.4250, 0.4140], std=[0.3512, 0.3312, 0.3284])
#     ])

# continuous_attr = ['top_curly','top']

class face_attributes(data.Dataset):
    # Root of single types of dataset
    def __init__(self, root, human_dir, transform=None, debug=False, target_mode='tag',train=True):
        self.root = root
        self.train = train
        self.human_dir = os.path.join(root, human_dir)
        self.asset_dir = os.path.join(root, 'asset')
        self.target_mode = target_mode
        
        self.transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]) if transform is None else transform
        self.debug = debug
        
        
        self.source_img_dir = os.path.join(self.human_dir, 'images')       # Human Images
        self.target_img_dir = os.path.join(self.human_dir, 'mapped_match') # Human mapped
        self.asset_img_dir  = os.path.join(self.asset_dir, 'images')       # Human assets

        self.source_img_paths = data_utils.make_image_set(self.source_img_dir)
        self.target_img_paths = data_utils.make_image_set(self.target_img_dir)
        self.source_img_paths,self.target_img_paths = sorted(self.source_img_paths),sorted(self.target_img_paths)
        assert len(self.source_img_paths) == len(self.target_img_paths) , 'source and target image number not match'

        # Get mathced image pairs names
        self.match_asset = [ path.split('/')[-1].split('.')[0].split('_')[-1] for path in self.target_img_paths]
        
        # load json file to dict
        self.asset_label_dict = json.load(open(os.path.join(self.asset_dir, 'soft_label.json')))
        self.source_img_label_dict = json.load(open(os.path.join(self.human_dir, 'soft_label.json')))
        # sort dict
        self.asset_label_dict = {k: v for k, v in sorted(self.asset_label_dict.items(), key=lambda item: item[0])}
        self.source_img_label_dict = {k: v for k, v in sorted(self.source_img_label_dict.items(), key=lambda item: item[0])}

        self.asset_label_dict,self.asset_label_cant_dict = self.process_soft_label(self.asset_label_dict)
        self.source_img_label_dict,self.source_img_label_cant_dict = self.process_soft_label(self.source_img_label_dict)


        if self.train:
            self.source_img_paths = self.source_img_paths[:int(len(self.source_img_paths)*0.8)]
            self.target_img_paths = self.target_img_paths[:int(len(self.target_img_paths)*0.8)]
            self.match_asset = self.match_asset[:int(len(self.match_asset)*0.8)]
        else:
            self.source_img_paths = self.source_img_paths[int(len(self.source_img_paths)*0.8):]
            self.target_img_paths = self.target_img_paths[int(len(self.target_img_paths)*0.8):]
            self.match_asset = self.match_asset[int(len(self.match_asset)*0.8):]

        # TODO Train/test
        for key in self.source_img_label_cant_dict:
            self.num_classes = len(self.source_img_label_cant_dict[key])
            break

    def __len__(self):
        return len(self.source_img_paths)
        
         
    def __getitem__(self, index):
        source_img_path = self.source_img_paths[index]
        source_img = self.transform(Image.open(source_img_path).convert('RGB'))
        source_img_name = source_img_path.split('/')[-1].split('.')[0]+'.png'

        if self.target_mode == 'tag':
            label = self.source_img_label_cant_dict[source_img_name]
            label = torch.FloatTensor(label)
        elif label.target_mode == 'img':
            label = self.match_asset[index] # TODO convert to one hot
            a=1
        return source_img, label,index
    
    def get_mean_std(self, img):
        self.mean = img.mean()
        self.std = img.std()
        return self.mean, self.std


    def process_soft_label(self, soft_label):
        # TODO
        aggre_label_all = {}
        for image in soft_label:
            aggre_label_all[image] = []
            label_dict = soft_label[image]

            for label in label_dict:
                a=1
                attr_dict = label_dict[label]
                vote_list = []
                for attr in attr_dict:
                    vote_list += [int(attr.split('-')[0])]*attr_dict[attr]
                average_vote = sum(vote_list)/len(vote_list)
                label_dict[label] = average_vote
                # if label == 'top_length':
                aggre_label_all[image].append(average_vote)
        return soft_label,aggre_label_all
    
    

if __name__ == '__main__':
    raw_root = '/home/minghao/data/navi_data/'
    human_dir = 'v3/'

    playground = 'playground/'
    class_type='braids_and_balls_train'

    train_dataset = face_attributes(raw_root,human_dir,debug=False,train=True)
    test_dataset = face_attributes(raw_root,human_dir,debug=False,train=False)
    # test_dataset  = face_attributes(raw_root,debug=False,class_type='braids_and_balls_test',train=False)
    print(len(train_dataset))
    # print(len(test_dataset))


    data_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(data_utils.get_mean_std_all(data_loader))


    for i, (img, label,idx) in enumerate(data_loader):
        print(img.shape)
        print(label)
        if i == 1:
            break

        # save_image(img, playground + 'img_' + str(i) + '.png')