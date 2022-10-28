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

# quality_info = ['blur','head_occlusion','mutiperson']


# continuous_attr = ['top_curly','top_length','side_curly','side_length']
# discrete_attr = ['braid_tf','braid_type','braid_count','braid_position']
# multi_class_attr = ['top_direction']

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
continuous_attr_dict = {
    'top_curly':4,
    'top_length':6,
    'side_curly':4,
    'side_length':5
}
discrete_attr_dict = {
    'braid_tf':2,
    'braid_type':5,
    'braid_count':4,
    'braid_position':3
}
multi_class_attr_dict = {
    'top_direction':8
}
class face_attributes(data.Dataset):
    # Root of single types of dataset

    #TODO: init too slow
    def __init__(self, root, human_dir, transform=None, debug=False, target_mode='tag',train_mode='train',debug_num=16):
        self.root = root
        self.train_mode = train_mode
        self.human_dir = os.path.join(root, human_dir)
        self.asset_dir = os.path.join(root, 'asset')
        self.target_mode = target_mode
        
        self.transform = transform  
        self.debug = debug

        target_dir = self.train_mode
        selected_cases = data_utils.make_image_set(os.path.join(self.human_dir, target_dir)) 
        selected_names = [x.split('/')[-1].split('.')[0] for x in selected_cases]
        if debug: selected_names = selected_names[:debug_num]
        
        
        self.source_img_dir = os.path.join(self.human_dir, 'images')       # Human Images
        self.target_img_dir = os.path.join(self.human_dir, 'mapped_match') # Human mapped
        self.asset_img_dir  = os.path.join(self.asset_dir, 'images')       # Human assets

        self.source_img_paths = data_utils.make_image_set(self.source_img_dir)
        self.target_img_paths = data_utils.make_image_set(self.target_img_dir)
        self.asset_img_paths  = data_utils.make_image_set(self.asset_img_dir)

        self.source_img_paths = self.select_case(self.source_img_paths, selected_names)
        self.target_img_paths = self.select_case(self.target_img_paths, selected_names) # Taske time in train init


        self.source_img_paths,self.target_img_paths = sorted(self.source_img_paths),sorted(self.target_img_paths)
        assert len(self.source_img_paths) == len(self.target_img_paths) , 'source and target image number not match'

        # Get mathced image pairs names
        self.match_asset = [ path.split('/')[-1].split('.')[0].split('_')[-1] for path in self.target_img_paths]
        
        self.asset_to_one_hot()

        # load soft label from json file to dict
        self.asset_label_dict = json.load(open(os.path.join(self.asset_dir, 'soft_label.json')))
        self.source_img_label_dict = json.load(open(os.path.join(self.human_dir, 'soft_label.json')))
        self.source_img_label_dict = self.select_case_dict(self.source_img_label_dict, selected_names)
        # sort label dict
        self.asset_label_dict = {k: v for k, v in sorted(self.asset_label_dict.items(), key=lambda item: item[0])}
        self.source_img_label_dict = {k: v for k, v in sorted(self.source_img_label_dict.items(), key=lambda item: item[0])}

        # process soft label
        self.asset_label_dict,self.asset_label_cant_dict = self.process_soft_label(self.asset_label_dict)
        self.source_img_label_dict,self.source_img_label_cant_dict = self.process_soft_label(self.source_img_label_dict) # Taske time in train init


        # if self.train:
        # self.source_img_paths = self.source_img_paths[:int(len(self.source_img_paths)*0.8)]
        # self.target_img_paths = self.target_img_paths[:int(len(self.target_img_paths)*0.8)]
        # self.match_asset = self.match_asset[:int(len(self.match_asset)*0.8)]
        # else:
        #     self.source_img_paths = self.source_img_paths[int(len(self.source_img_paths)*0.8):]
        #     self.target_img_paths = self.target_img_paths[int(len(self.target_img_paths)*0.8):]
        #     self.match_asset = self.match_asset[int(len(self.match_asset)*0.8):]

        # TODO Train/test
        if self.target_mode =='tag':
            for key in self.source_img_label_cant_dict:
                self.num_classes = len(self.source_img_label_cant_dict[key])
                break
        elif self.target_mode =='img':
            self.num_classes = len(self.asset_img_paths)

    # TODO fix one hot order
    def asset_to_one_hot(self):
        self.all_asset_names = [path.split('/')[-1].split('.')[0] for path in self.asset_img_paths]

        self.match_asset_one_hot = []
        for i,asset_name in enumerate(self.match_asset):
            one_hot = [0]*len(self.all_asset_names)
            one_hot[self.all_asset_names.index(asset_name)] = 1
            # one_hot to np array
            self.match_asset_one_hot.append(np.array(one_hot))

        a=1

        
    def __len__(self):
        return len(self.source_img_paths)
        
         
    def __getitem__(self, index, extra_info=False):
        source_img_path = self.source_img_paths[index]
        source_img = self.transform(Image.open(source_img_path).convert('RGB'))
        source_img_name = source_img_path.split('/')[-1].split('.')[0]#+'.png'

        if self.target_mode == 'tag':
            label = self.source_img_label_cant_dict[source_img_name]
            label = torch.FloatTensor(label)
        elif self.target_mode == 'img':
            # label = self.match_asset[index] # TODO convert to one hot
            label = self.match_asset_one_hot[index]
            label = torch.FloatTensor(label)
        
        return source_img, label, index

    def get_index_dict(self):
        return self.index_dict
    
    def get_mean_std(self, img):
        self.mean = img.mean()
        self.std = img.std()
        return self.mean, self.std


    def process_soft_label(self, soft_label):
        # TODO
        # continuous_attr_dict,discrete_attr_dict,multi_class_attr_dict
        total_param = 0
        index_dict = {}
        for key in continuous_attr_dict:
            index_dict[key] = [total_param,total_param]
            total_param += 1
        for key in discrete_attr_dict:
            index_dict[key] = [total_param,total_param+discrete_attr_dict[key]-1]
            total_param += discrete_attr_dict[key]
        for key in multi_class_attr_dict:
            index_dict[key] = [total_param,total_param+multi_class_attr_dict[key]-1]
            total_param += multi_class_attr_dict[key]
        template_param = [0]*total_param
            

        aggre_label_all = {}
        for image in soft_label:
            aggre_label_all[image] = template_param.copy()
            label_dict = soft_label[image]

            for label in label_dict:

                attr_dict = label_dict[label]

                # Convert vote for string dict to int list
                vote_list = []
                for attr in attr_dict:
                    vote_list += [int(attr.split('-')[0])]*attr_dict[attr]

                # Process continuous attr (average)
                if label in continuous_attr_dict:
                    aggre_vote = np.mean(vote_list)
                    aggre_label_all[image][index_dict[label][0]] = aggre_vote
                    a=1

                # Process discrete attr (vote)
                elif label in discrete_attr_dict:
                    soft = self.soft_label_to_one_hot(vote_list,discrete_attr_dict[label])
                    aggre_label_all[image][index_dict[label][0]:index_dict[label][1]+1] = soft
                    a=1
                
                # Process multi class attr (vote)
                elif label in multi_class_attr_dict:
                    soft = self.soft_label_to_one_hot(vote_list,multi_class_attr_dict[label])

                    aggre_label_all[image][index_dict[label][0]:index_dict[label][1]+1] = soft
                    a=1
                
                assert len(aggre_label_all[image]) == total_param, 'Soft label length not match'
                # average_vote = sum(vote_list)/len(vote_list)
                # label_dict[label] = average_vote
                # if label == 'top_length':
                # aggre_label_all[image].append(average_vote)
        self.index_dict = index_dict
        return soft_label,aggre_label_all

    # soft max of aggregated one hot vectors
    def soft_label_to_one_hot(self, vote_list, num_classes):
        one_hots = self.one_hot(np.array(vote_list),num_classes) # Get one hot
        sum_  = np.sum(one_hots,axis=0) # Aggregate votes by sum
        soft  = np.exp(sum_)/sum(np.exp(sum_)) # Use softmax to get soft label
        soft  = soft.tolist()
        
        return soft

    def one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    def select_case(self, all_paths, selected_names):
        selected_out = []
        for item in all_paths:
            cur = item.split('/')[-1].split('.')[0]
            cur = cur.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','')
            cur = cur.split('_')[0]
            if cur in selected_names:
                selected_out.append(item)
        return selected_out
    
    def select_case_dict(self, label_dict, selected_names):
        out_dict = {}
        for key in label_dict:
            cur = key.split('/')[-1].split('.')[0]
            cur = cur.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','')
            cur = cur.split('_')[0]

            if cur in selected_names:
                out_dict[cur] = label_dict[key]

        return out_dict



if __name__ == '__main__':
    raw_root = '/media/otter/navi_hitl/'
    human_dir = 'FairFace2.0/'

    # playground = 'playground/'
    # class_type='braids_and_balls_train'

    train_dataset = face_attributes(raw_root,human_dir,debug=True,train_mode='train')
    
    # test_dataset = face_attributes(raw_root,human_dir,debug=False,train=False)
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