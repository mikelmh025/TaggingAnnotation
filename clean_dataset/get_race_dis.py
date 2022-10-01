import os
import os.path as osp
import data_utils
import cv2
from pathlib import Path

import numpy as np
import csv

# from py_utils.file_system.file_utils import check_path


label_attris = ['race']
data_types = ['clean','blur','hat','multiple','no_person','original']


dataset_root = '/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/'
dataset_name = 'train'
dataset_dir  =dataset_root + dataset_name


csv_type = 'train' if 'train' in dataset_dir.split('/')[-1]  else 'val'
csv_path = '/Users/bytedance/Desktop/data/image datasets/fairface_label_%s.csv'%csv_type

# Read CSV as dict
csv_dict = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['file'].replace(csv_type+'/','')
        csv_dict[name] = {}
        for label_attri in label_attris:
            csv_dict[name][label_attri] = row[label_attri]



img_paths = data_utils.make_im_set(dataset_dir)
result_dict={}
for data_type in data_types: 
    result_dict[data_type] = {}
    for label_attri in label_attris: result_dict[data_type][label_attri] = {}

# Get stats for each image
for img_path in img_paths:
    name = img_path.split('/')[-1]
    # based on the dataset types
    for data_type in data_types:
        cur_data_path = dataset_root+csv_type+'_'+data_type+'/'+name if data_type != 'original' else dataset_root+csv_type+'/'+name
        if osp.exists(cur_data_path):
            label = csv_dict[name]
            for label_attri in label_attris:
                if label[label_attri] in result_dict[data_type][label_attri] :
                    result_dict[data_type][label_attri][label[label_attri]] += 1
                else:
                    result_dict[data_type][label_attri][label[label_attri]] = 1 

def get_percentage_sort(dict,dict_original):
    sort_key_map ={} 
    for key in dict:
        if '-' in key:
            sort_key_map[int(key.split('-')[0])] = key
        else:
            digit = ''.join(x for x in key if x.isdigit())
            if digit != '':
                sort_key_map[int(digit)] = key
            else:
                sort_key_map[key] = key

    sorted_keys, sorted_vals, sorted_vals_original = [], [], []
    for sort_key in sorted(sort_key_map):
        sorted_keys += [sort_key_map[sort_key]]
        sorted_vals += [dict[sort_key_map[sort_key]]]
        sorted_vals_original += [dict_original[sort_key_map[sort_key]]]

        # print (sort_key_map[sort_key],val)
    sorted_vals = [round(val/sum(sorted_vals)*100,2) for val in sorted_vals]
    sorted_vals_original = [round(val/sum(sorted_vals_original)*100,2) for val in sorted_vals_original]
    val_diff = [round(val-val_original,2) for val,val_original in zip(sorted_vals,sorted_vals_original)]
    return sorted_keys, sorted_vals, val_diff
    

# Write result_dict to csv
with open('/Users/bytedance/Desktop/data/image datasets/fairface_label_%s_stats.csv'%csv_type, 'w') as f:
    writer = csv.writer(f)
    for data_type in data_types:
        writer.writerow(['Dataset type',csv_type+'_'+data_type])
        for label_attri in label_attris:
            label_class, label_perc, val_diff = get_percentage_sort(result_dict[data_type][label_attri],result_dict['original'][label_attri])
            # label_class = list(result_dict[data_type][label_attri].keys())
            # label_count = list(result_dict[data_type][label_attri].values())
            writer.writerow(["label_attri : " + label_attri]+label_class)
            # writer.writerow(["%s_count :"%label_attri]+label_count)
            writer.writerow(["%s_perc :" %label_attri]+label_perc )
            writer.writerow(["%s_val_diff :" %label_attri]+val_diff )

            
        writer.writerow([''])

    f.close()
        
        