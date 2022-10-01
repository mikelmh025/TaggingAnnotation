import csv
import numpy as np
import torch
import torch.nn as nn
import data_utils
import os
import math
import process_utils
from pathlib import Path
import json

from PIL import Image
import requests
from io import BytesIO

root = '/Users/bytedance/Desktop/data/annotations/'
csv_path = root + '721_fairface_face_Info_1225.csv'
save_json = True
get_image = False
save_aggre_json=True
save_clean_only=True
annotation_round = 3

prefix = 'hair_sub_attribute_0713_'
prior_root='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/annotation_degbug/'

prior_labels = process_utils.prior_labels
target_label = process_utils.target_label
base_row_titles = process_utils.base_row_titles
base_dict_titles = process_utils.base_dict_titles
round_base_row_titles  = process_utils.round_base_row_titles
round_base_dict_titles = process_utils.round_base_dict_titles
round_quality_titles = process_utils.round_quality_titles
round_attri_one_titles = process_utils.round_attri_one_titles
round_attri_two_titles = process_utils.round_attri_two_titles
round_row_titles  = process_utils.round_row_titles
round_dict_titles = process_utils.round_dict_titles
round_dict_titles_GUI = process_utils.round_dict_titles_GUI
terms_val_mapper = process_utils.terms_val_mapper
string2int = process_utils.string2int
terms_val_mapper_GUI = process_utils.terms_val_mapper_GUI
round_quality_titles_GUI = process_utils.round_quality_titles_GUI

data_dict = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_dict = {}
        # Get basic info of the task
        for base_row_title, base_dict_title in zip(base_row_titles, base_dict_titles):
            row_dict[base_dict_title] = row[base_row_title]

        # Get annotion labels from each round of annotation
        for round in range(annotation_round):
            round_name = str(round+1)+"_"
            for round_row_title, round_dict_title in zip(round_row_titles, round_dict_titles):
                # Translate from string annotaiton to int value
                if round_dict_title in terms_val_mapper_GUI:
                    row_dict[round_name+round_dict_title] = terms_val_mapper_GUI[round_dict_title][row[round_name+round_row_title]]
                if round_dict_title in round_quality_titles_GUI:
                    row_dict[round_name+round_dict_title] = terms_val_mapper[round_dict_title][row[round_name+round_row_title]]
                    
        file_name = row_dict['image_url'].split('/')[-1].replace('\t', '')
        data_dict[file_name] = row_dict

        a=1


output_json_dict = {}
for task_id in data_dict:
    output_json_dict[task_id] = {}
    task_dict = data_dict[task_id]

    # Check the qulity of the image 
    quality_dict, cur_clean,save_curr = {}, True, True
    # for round in range(annotation_round):
    #     round_name = str(round+1)
    #     # Save quality of each round
    #     for round_row_title, round_dict_title,round_dict_title_GUI in zip(round_row_titles, round_dict_titles, round_dict_titles_GUI):
    #         if round_dict_title in round_quality_titles:
    #             if round_dict_title not in quality_dict:
    #                 quality_dict[round_dict_title] = [task_dict[round_name+"_"+round_dict_title]]
    #             else:
    #                 quality_dict[round_dict_title].append(task_dict[round_name+"_"+round_dict_title])
    # for key in quality_dict:
    #     quality_dict[key] = max(set(quality_dict[key]), key = quality_dict[key].count)
    #     if quality_dict[key] != 2: cur_clean = False
    cur_clean = task_dict['image_url'].replace('\t', '').split('/')[-1].split('.')[0].split('_')[-1] == '1'

    if save_clean_only and not cur_clean: save_curr = False

    # Save annotation of each round to json
    for round in range(annotation_round):
        round_name = str(round+1)
        round_dict = {}
        braid = task_dict[round_name+"_"+'braid'] == '0-Braid' # Special case
        for round_row_title, round_dict_title,round_dict_title_GUI in zip(round_row_titles, round_dict_titles, round_dict_titles_GUI):
            # Translate from string annotaiton to int value
            if round_dict_title in terms_val_mapper_GUI:
                save_val = task_dict[round_name+"_"+round_dict_title]
                round_dict[round_dict_title_GUI] = [save_val] if isinstance(save_val, str) else save_val
                
                if round_dict_title_GUI == 'texture' and braid: round_dict[round_dict_title_GUI].append('7-braids')
            
            
        a=1

        output_json_dict[task_id][round_name] = round_dict

        if save_json and save_curr:
                
            save_dir = root + 'round' + round_name
            file_name = task_id.split('/')[-1].replace('\t', '').split('.')[0]
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            with open(save_dir + '/' + file_name+ '.json', 'w') as f:
                json.dump(round_dict, f)

    # Save image from URL
    if get_image and save_curr:
        url = task_dict['image_url'].replace('\t', '')
        image_dir = root + 'image'
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        image_path = image_dir + '/' + task_id
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(image_path)
    
    aggre_dict = {}
    for key in output_json_dict[task_id]['1']:
        aggre_dict[key] = []
        for round in range(annotation_round):
            round_name = str(round+1)
            val = output_json_dict[task_id][round_name][key]
            if len(val) > 1:
                if key == 'bang':
                    val = ['1-little']
                elif key == 'texture':
                    val.remove('7-braids')
            aggre_dict[key] += val


    for key in aggre_dict:
        assert len(aggre_dict[key]) == annotation_round
        annotation_list = aggre_dict[key]
        aggregated_val  = [max(set(annotation_list), key=annotation_list.count)]
        if key == 'bang' and aggregated_val == '1-little': 
            aggregated_val = ['1-little','2-parting']
        if key == 'texture' and aggre_dict['If braid'] == ['0-Braid']:
            aggregated_val += ['7-braids']
        aggre_dict[key] = aggregated_val

    if save_aggre_json and save_curr:
        save_dir = root + 'aggregated' + str(annotation_round)
        file_name = task_id.split('/')[-1].replace('\t', '').split('.')[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(save_dir + '/' + file_name+ '.json', 'w') as f:
            json.dump(round_dict, f)
    a=1

a=1
