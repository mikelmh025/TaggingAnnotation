import csv
import json
import data_utils
from PIL import Image
import cv2
import numpy as np
import os
import random

root = '/Users/bytedance/Desktop/data/annotations/0_annotation/FairFace2.0/'

name = 'all_results'
csv_path = root+name+'.csv'
# csv_path = root+'5k_results.csv'

quality_check_csv_path = root+ name + '_quality_check_all.csv'
quality_check_10_csv_path = root+ name + '_quality_check_10.csv'

csv_path2 = root+'829_faceattribute_round2_asset_translate.csv'
strict_json_path = csv_path.replace('.csv', '_strict.json')
relax_json_path = csv_path.replace('.csv', '_relax.json')
soft_json_path = csv_path.replace('.csv', '_soft.json')


annotation_round = 3
annotation_round2 = 0 #6



basic_info = ['\ufeffTask ID','Object ID']
round_info = ['Handling Time','Record','Moderator']

quality_info = ['blur','head_occlusion','mutiperson']
top_info = ['top_curly','top_direction','top_length']
side_info = ['side_curly','side_length']
braid_info = ['braid_tf','braid_type','braid_count','braid_position']

multi_select_attributes  = ['top_direction','braid_position']
top_direction_group1 = ['0-向下','1-斜下','2-横向','3-斜上','6-向上（头发立起来）','7-向后梳头（长发向后梳，大背头，马尾）']
top_direction_group2 = ['4-中分','5-37分']

record_info = quality_info + top_info + side_info + braid_info

def load_record(row,round_name):
    info_list = json.loads(row[round_name+'Record'])['mainForm']
    out_dict = {}
    for info in info_list:
        if info['name'] not in record_info:continue
        var_name = info['name']
        value    = info['value']
        # out_dict[var_name] = [int(value[0])] if var_name not in out_dict else out_dict[var_name] + [int(value[0])]
        out_dict[var_name] = [value] if var_name not in out_dict else out_dict[var_name] + [value]

        
    return out_dict

def csv2datadict(csv_path,data_dict = {},annotation_round = 3, round_start=0):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_dict = {}

            for key in basic_info:
                try:
                    row_dict[key.replace('\ufeff', '')] = row[key].replace('\t', '')
                except:
                    row_dict[key.replace('\ufeff', '')] = row[key.replace('\ufeff', '')].replace('\t', '')

            
            row_dict['task_name']    = row['Object ID'].split('/')[-1].split('.')[0]
            row_dict['task_image_name'] = row_dict['task_name'].replace('820_faceattribute_round2_','')

            for i in range(annotation_round):
                round_name = str(i+1)+"Round"
                record_name = str(i+1+round_start)+"Round"
                for key in round_info:
                    if 'Record' in key:
                        row_dict[record_name+key] = load_record(row,round_name)#json.loads(row[round_name+key])
                    else:
                        if 'Time' in key:
                            row_dict['Time'+str(i+1+round_start)] = row[round_name+key]
                        elif 'Moderator' in key:
                            row_dict['Moderator'+str(i+1+round_start)] = row[round_name+key]
                        else:
                            row_dict[record_name+key] = row[round_name+key]
            if row_dict['task_name'] not in data_dict:
                data_dict[row_dict['task_name']] = row_dict
            else:
                for item in row_dict:
                    if item not in data_dict[row_dict['task_name']]:
                        data_dict[row_dict['task_name']][item] = row_dict[item]
                    else:
                        continue
                a=1
    return data_dict

data_dict = {}
data_dict = csv2datadict(csv_path ,data_dict,annotation_round)
# data_dict = csv2datadict(csv_path2,data_dict,annotation_round2,round_start=annotation_round)
annotation_round = annotation_round + annotation_round2
a=1


def multi_val(val_list,attribute_type):

    care = 0
    if attribute_type == 'braid_position':
        # TODO assert 辫子TF
        soft_label = {}
        
        for idx, item in enumerate(val_list):
            if item == ['1-高辫子（如高马尾）', '2-低辫子'] or item ==['2-低辫子', '1-高辫子（如高马尾）']:
                val_list[idx] = '9-Not sure'
                for i in item:
                    soft_label[i] = 1 if i not in soft_label else soft_label[i] + 1
            elif isinstance(item, list):
                for i in item:
                    soft_label[i] = 1 if i not in soft_label else soft_label[i] + 1
                val_list[idx] = '9-Not sure'
            else:
                soft_label[item] = 1 if item not in soft_label else soft_label[item] + 1

        # if len(set(set(val_list))) == 3:
        #     aggre = ['9-Not sure']
        else:
            aggre = [max(set(val_list), key=val_list.count)] 
            
        # change from  9-Not sure to ['1-高辫子（如高马尾）', '2-低辫子']
        aggre = ['0-没有辫子','1-高辫子（如高马尾）', '2-低辫子'] if aggre == '9-Not sure' else aggre
        strict_aggre, relax_aggre = aggre,aggre
        
    elif attribute_type == 'top_direction':
        
        cur_sub_group1 = []
        cur_sub_group2 = []
        soft_label = {}

        def add_to_group(val):
            if val in top_direction_group1:
                cur_sub_group1.append(val)
            else:
                cur_sub_group2.append(val)

        for idx, item in enumerate(val_list):
            if isinstance(item,list):
                # assert len(item) == 2
                # if item[0] in top_direction_group1 and item[1] in top_direction_group1: care = 1
                # if item[0] in top_direction_group2 and item[1] in top_direction_group2: care = 1

                for i in item: add_to_group(i)
                
            else:
                add_to_group(item)
        aggre = list(set(cur_sub_group1)) + list(set(cur_sub_group2) )
        relax_aggre,strict_aggre = aggre,aggre

        # create soft label
        for item in cur_sub_group1+cur_sub_group2:
            soft_label[item] = 1 if item not in soft_label else soft_label[item] + 1

        if len(set(cur_sub_group1)) >= 3 or len(set(cur_sub_group2)) >= 3:
            care = 1

        # When only have two agreement out of 3, strict. 
        if len(set(cur_sub_group1)) == 2 or len(set(cur_sub_group2)) == 2:
            temp_1 = [max(set(cur_sub_group1), key=cur_sub_group1.count)]  if len(set(cur_sub_group1)) == 2 else  list(set(cur_sub_group1) )
            temp_2 = [max(set(cur_sub_group2), key=cur_sub_group2.count)]  if len(set(cur_sub_group2)) == 2 else  list(set(cur_sub_group2) )
            strict_aggre = temp_1 + temp_2
            care = 2

    return strict_aggre, relax_aggre, soft_label, care

total_counter,multi_val_counter,single_val_counter = 0, 0, 0
single_val_two_counter,multi_val_two_counter = 0, 0

strict_aggre_json_dict = {}
relax_aggre_json_dict  = {}
soft_aggre_json_dict = {}

check_data_dict = {}
for task in data_dict:
    image_name = data_dict[task]['task_image_name'] +'.png'

    check_data_dict [image_name] = {}
    check_data_dict [image_name]['Task ID'] = data_dict[task]['Task ID']
    check_data_dict [image_name]['url'] = data_dict[task]['Object ID']


    cur_dict = {}
    strict_aggre_dict = {}
    relax_aggre_dict = {}
    soft_label_dict = {}
    for attribute in record_info:
        flag_multi_value = attribute in multi_select_attributes
        cur_dict[attribute] = []
        for i in range(annotation_round):
            round_name = str(i+1)+"Round"+"Record"
            cur_record = data_dict[task][round_name]
            if len(cur_record[attribute]) == 1:
                cur_dict[attribute] += cur_record[attribute]
            else:
                cur_dict[attribute] += [cur_record[attribute]]

            for item in cur_record:
                check_data_dict[image_name][item+str(i+1)] = str(cur_record[item][0][0])
            check_data_dict[image_name]['Time'+str(i+1)] = str(data_dict[task]['Time'+str(i+1)])
            check_data_dict[image_name]['Moderator'+str(i+1)] = str(data_dict[task]['Moderator'+str(i+1)])
        

        #TODO: deal with braid_position, top_direction
        # braid_position: relax, based on TF also
        # top_direction:  aggregate based on sub-groups also. 
        if (flag_multi_value):
            strict_aggregated_val, relax_aggregated_val,soft_label,care = multi_val(cur_dict[attribute],attribute)
            if care==1:
                multi_val_counter += 1
            elif care==2:
                multi_val_two_counter += 1
            # soft_label = cur_dict[attribute]
        else:
            relax_aggregated_val, strict_aggregated_val = list(set(cur_dict[attribute])),  list(set(cur_dict[attribute]))
            soft_label = cur_dict[attribute]
            soft_label = {}
            for item in relax_aggregated_val: soft_label[item] = 0
            for item in cur_dict[attribute]:  soft_label[item] += 1


            if len(set(cur_dict[attribute])) == 2:
                strict_aggregated_val = [max(set(cur_dict[attribute]), key=cur_dict[attribute].count)]
                single_val_two_counter += 1
            elif len(set(cur_dict[attribute])) == 3:
                single_val_counter += 1
                
        strict_aggre_dict[attribute] = strict_aggregated_val
        relax_aggre_dict[attribute]  = relax_aggregated_val
        soft_label_dict[attribute]   = soft_label
        total_counter += 1

    strict_aggre_json_dict[image_name] = strict_aggre_dict
    relax_aggre_json_dict[image_name]  = relax_aggre_dict
    soft_aggre_json_dict[image_name]   = soft_label_dict

a=1


print('total_counter:', total_counter)
print('multi_val_counter:', multi_val_counter)
print('single_val_counter:', single_val_counter)

print('single_val_two_counter:', single_val_two_counter)
print('multi_val_two_counter:', multi_val_two_counter)

# save to json 
with open(strict_json_path, 'w') as f:
    json.dump(strict_aggre_json_dict, f)

# save to json 
with open(relax_json_path, 'w') as f:
    json.dump(relax_aggre_json_dict, f)

# save to json
with open(soft_json_path, 'w') as f:
    json.dump(soft_aggre_json_dict, f)

# Write to CSV
for key in check_data_dict:
    header = check_data_dict[key].keys()
    break


with open(quality_check_csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for key in check_data_dict:
        writer.writerow(check_data_dict[key].values())
        # writer.writerow([url])

with open(quality_check_10_csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    # random select 10 from check_data_dict
    for key in random.sample(check_data_dict.keys(), int(len(check_data_dict)*0.1)):
        writer.writerow(check_data_dict[key].values())

