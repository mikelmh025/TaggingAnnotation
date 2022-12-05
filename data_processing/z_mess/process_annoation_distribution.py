import csv
import numpy as np
import torch
import torch.nn as nn
import data_utils
import os
import math

import process_utils

# Read CSV file 
csv_path = '/Users/bytedance/Desktop/data/annotations/721_fairface_face_Info_1225.csv'
csv_path_out = csv_path.replace('.csv','_out.csv')
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
terms_val_mapper = process_utils.terms_val_mapper
string2int = process_utils.string2int


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
                if round_dict_title in terms_val_mapper:
                    row_dict[round_name+round_dict_title] = terms_val_mapper[round_dict_title][row[round_name+round_row_title]]
                elif round_dict_title in string2int:
                    row_dict[round_name+round_dict_title] = int(row[round_name+round_row_title])
                else:
                    row_dict[round_name+round_dict_title] = row[round_name+round_row_title]
        data_dict[row_dict['id']] = row_dict

        a=1
    # data = list(reader['url'])


# Raw data_dict to data_dict_by_task
data_dict_by_task = {}
for task_id in data_dict:
    task_dict = data_dict[task_id]
    # Get annotion labels from each round of annotation
    task_base_dict = {}
    task_annotor_dict = {}
    task_result_dict = {}
    task_entropy_dict = {}

    # TODO based on image name find prior labels
    prior_label_dict = {}
    for item in task_dict['image_url'].split('/'):
        if data_utils.is_image_file(item.replace('\t','')): image_name = item.replace('\t','')

    image_name = image_name.replace('hair_sub_attribute_0713_','')
    for idx, prior_label in  enumerate(prior_labels):
        prior_label_path = prior_root+prior_label+'/'+image_name
        prior_label_dict[prior_label] = True if os.path.exists(prior_label_path) else False
    
    for val_name in task_dict:
        # Get base info of the task
        if val_name in base_dict_titles:
            task_base_dict[val_name] = task_dict[val_name]
        else:                
            # name_ = val_name.split('_')[0]
            name_ = val_name.replace(val_name.split('_')[0]+"_",'')
            if name_ in round_base_dict_titles:
                task_annotor_dict[name_] = [task_dict[val_name]] if name_ not in task_annotor_dict else task_annotor_dict[name_] + [task_dict[val_name]]
            elif name_ in terms_val_mapper:
                # if name_ == 'curly':
                #     a=1
                #     print('val_name',val_name)
                task_result_dict[name_] = [task_dict[val_name]] if name_ not in task_result_dict else task_result_dict[name_] + [task_dict[val_name]]

    # Get entropy of each attributes
    # for key in task_result_dict: task_entropy_dict[key] = get_entropy(task_result_dict[key],len(terms_val_mapper[key]))
    data_dict_by_task[task_id] = {'base':task_base_dict,'annotor':task_annotor_dict,'result':task_result_dict,'prior_label':prior_label_dict} 


# analysis case by case
task_entropy = []
task_duration = []
attribute_summary_dict = {}
for key in terms_val_mapper: 
    attribute_summary_dict[key] = {}
    attribute_summary_dict[key]['aggre_count'] = [0 for i in range(len(terms_val_mapper[key]))]
    attribute_summary_dict[key]['entro']       = []
    attribute_summary_dict[key]['ce_aggre']    = []
    attribute_summary_dict[key]['acc']    = []
    attribute_summary_dict[key]['mse']    = []
    attribute_summary_dict[key]['mse_wrong']= []
    

for task_id in data_dict_by_task:

    # Filter out by prior label
    if target_label!= None and data_dict_by_task[task_id]['prior_label'][target_label] == False: continue
            


    # Get entropy
    entropy_list = []
    entropy_dict = {}
    for key in task_result_dict: entropy_dict[key] = data_utils.get_entropy(data_dict_by_task[task_id]['result'][key],len(terms_val_mapper[key]))  # Create dict version
    for key in entropy_dict: entropy_list += [entropy_dict[key]]                  # Create List version

    average_entropy = np.mean(entropy_list)
    task_entropy += [average_entropy]

    # Get duration
    duration = data_dict_by_task[task_id]['annotor']['duration']
    average_duration = np.mean(duration)
    task_duration += [average_duration]
    a=1

    # Get attribute summary
    for key in attribute_summary_dict:
        if key == 'hairstyle_curly':
            a=1
        prediction = data_dict_by_task[task_id]['result'][key]
        aggregate  = max(set(prediction), key=prediction.count)

        # compute cross entropy from prediction and aggregate
        loss = data_utils.get_cross_entropy(prediction,aggregate,len(terms_val_mapper[key])+1)
        acc  = data_utils.get_accuracy(prediction,aggregate)
        mse  = data_utils.get_mse(prediction,aggregate)

        attribute_summary_dict[key]['entro']                  += [entropy_dict[key]]
        attribute_summary_dict[key]['aggre_count'][aggregate] += 1
        attribute_summary_dict[key]['ce_aggre']               += [loss]
        attribute_summary_dict[key]['acc']                    += [acc]
        attribute_summary_dict[key]['mse']                    += [mse]
        attribute_summary_dict[key]['mse_wrong']              += [mse/(1-acc+1e-10)]
        a=1

for key in attribute_summary_dict:
    attribute_summary_dict[key]['entro']     = np.mean(attribute_summary_dict[key]['entro'])
    attribute_summary_dict[key]['ce_aggre']  = np.mean(attribute_summary_dict[key]['ce_aggre'])
    attribute_summary_dict[key]['acc']       = np.mean(attribute_summary_dict[key]['acc'])
    attribute_summary_dict[key]['mse']       = np.mean(attribute_summary_dict[key]['mse'])
    attribute_summary_dict[key]['mse_wrong'] = math.sqrt(attribute_summary_dict[key]['mse']/(1-attribute_summary_dict[key]['acc']+1e-10))
a=1


# Write to CSV file
with open(csv_path_out, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['attribute_type','entropy','aggregate_count','ce_aggre','acc','mse','mse_wrong'])
    for key in attribute_summary_dict:
        writer.writerow([key,attribute_summary_dict[key]['entro'],attribute_summary_dict[key]['aggre_count'], \
        attribute_summary_dict[key]['ce_aggre'],attribute_summary_dict[key]['acc'],attribute_summary_dict[key]['mse'],attribute_summary_dict[key]['mse_wrong']])