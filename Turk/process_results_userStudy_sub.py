import json
import data_utils
import os
import cv2
from tqdm import tqdm
import csv
import numpy as np
import copy



root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/user_study_run1/'
label_csv = root + 'subjective_run1.csv'



attri_need = {
    '1':['Input.image_person1',	'Input.image_cartoon1'],
    '2':['Input.image_person2',	'Input.image_cartoon2'],
    '3':['Input.image_person3',	'Input.image_cartoon3'],	
    'out': ['Answer.result']
}



attri_need_idx = {}
for item in attri_need:
    attri_need_idx[item] = [0]*len(attri_need[item])

# Load csv file
def load_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

rows = load_csv(label_csv)

data_dict = {}
header = rows[0]



for idx, item in enumerate(header):
    for key in attri_need:
        if item in attri_need[key]:
            # find the index of item in attri_need[key]
            index = attri_need[key].index(item)
            attri_need_idx[key][index] = idx



def string2dict(string):
    string = string.replace('\'','')
    string = string.replace(' ','')
    string = string.replace('True','true')
    string = string.replace('False','false')
    string = string.replace('None','null')

    return json.loads(string)

# debug_list = []
# TODO: deal with 3 votes

correct_dict = {}
correct = [0,0]
for row_idx, row in enumerate(rows):
    if row_idx==0:
        continue

    
    for vote in attri_need_idx:
        if vote == 'out': continue
        
        input_human_ = row[attri_need_idx[vote][0]]
        input_name = input_human_.split('/')[-1].split('.')[0]


        input_options_ = row[attri_need_idx[vote][1]:attri_need_idx[vote][-1]+1]
        method_name = input_options_[0].split('/')[-3:-1]
        method_name = '_'.join(method_name)

        # Process annotated results
        output_result = row[attri_need_idx['out'][0]]
        output_result_ = string2dict(output_result)['group'+vote]


        if method_name not in correct_dict:
            correct_dict[method_name] = [0,0]

        if output_result_ == '1':
            correct_dict[method_name] = [correct_dict[method_name][0]+1, correct_dict[method_name][1]+1]
        elif output_result_ == '0':
            correct_dict[method_name] = [correct_dict[method_name][0], correct_dict[method_name][1]+1]
            # if method_name == 'test_tag_bd_1':
            #     debug_list.append(int(input_name))
        else:
            continue
            # print('error')

a=1
# sort correct_dict by key
correct_dict = dict(sorted(correct_dict.items(), key=lambda item: item[0]))

for key in correct_dict:
    print(key, round(100*correct_dict[key][0]/correct_dict[key][1],2))

# debug_list.sort()
# print(debug_list)