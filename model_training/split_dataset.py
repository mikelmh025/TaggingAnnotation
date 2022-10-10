import data_utils   
import numpy as np
import csv
import json
import os
import cv2

# set np seed
np.random.seed(233)

image_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/images'
label_csv  =  '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/fairface_label_train.csv'

image_paths = data_utils.make_im_set(image_dir)


# Load csv file
with open(label_csv, 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

ethic_dict = {}
data_dict = {}
for image_path in image_paths:
    idx = int(image_path.split('/')[-1].split('.')[0].split('_')[-1])
    label = rows[idx]
    assert label[0].split('/')[-1].split('.')[0] == str(idx)

    data_dict[idx] = label + [image_path]
    if label[3] not in ethic_dict:
        ethic_dict[label[3]] = [str(idx)+'.jpg']
    else:
        ethic_dict[label[3]] += [str(idx)+'.jpg']

ethic_full_dict = {}
for row in rows:
    if row[3] not in ethic_full_dict:
        ethic_full_dict[row[3]] = [row[0]]
    else:
        ethic_full_dict[row[3]] += [row[0]]

# Split dataset into train, val, test
# 0.85, 0.13, 0.02
test_ratio = 0.012
val_ratio = 0.148
train_ratio = 0.85


train_dict = {}
val_dict = {}
test_dict = {}

for key in ethic_dict:
    # Shuffle image list
    np.random.shuffle(ethic_dict[key])
    # Split dataset
    test_dict[key] = ethic_dict[key][:int(len(ethic_dict[key])*test_ratio)]
    val_dict[key] = ethic_dict[key][int(len(ethic_dict[key])*test_ratio):int(len(ethic_dict[key])*test_ratio)+int(len(ethic_dict[key])*val_ratio)]
    train_dict[key] = ethic_dict[key][int(len(ethic_dict[key])*test_ratio)+int(len(ethic_dict[key])*val_ratio):]


def print_distribution_distance(data_dict1, data_dict2, count1, count2):
    for key in data_dict1:
        print(key, len(data_dict1[key])/count1 - len(data_dict2[key])/count2)

def print_distribution(data_dict,total_count):
    for key in data_dict:
        print(key, len(data_dict[key])/total_count)

def count_total(dict):
    count = 0
    for key in dict:
        count += len(dict[key])
    return count

def unique_dict(dict1, dict2):
    for key in dict1:
        for item in dict1[key]:
            if item in dict2[key]:
                return False
    return True

assert unique_dict(train_dict, val_dict)
assert unique_dict(train_dict, test_dict)
assert unique_dict(val_dict, test_dict)
assert count_total(train_dict) + count_total(val_dict) + count_total(test_dict) == len(image_paths)


#  Show distance between train, val, test and original full 100k fair face dataset
print("train count", count_total(train_dict))
print("train distribution")
# print_distribution(train_dict, count_total(train_dict))
print_distribution_distance(train_dict, ethic_full_dict, count_total(train_dict), len(rows))
print(" ------ ")

print("val distribution")
# print_distribution(val_dict, count_total(val_dict))
print_distribution_distance(val_dict, ethic_full_dict, count_total(val_dict), len(rows))
print("val count", count_total(val_dict))
print(" ------ ")

print("test distribution")
# print_distribution(test_dict, count_total(test_dict))
print_distribution_distance(test_dict, ethic_full_dict, count_total(test_dict), len(rows))
print("test count", count_total(test_dict))
print(" ------ ")

print("overall distribution")
# print_distribution(ethic_dict, len(image_paths))
print_distribution_distance(ethic_dict, ethic_full_dict, len(image_paths), len(rows))
print("overall count", len(image_paths))
print(" ------ ")

print("full distribution")
# print_distribution(ethic_full_dict, len(rows))
print_distribution_distance(ethic_full_dict, ethic_full_dict, len(rows), len(rows))
print("full count", len(rows))
print(" ------ ")

a=1

def save_dict_2_json(path_dict,label_dict,root,name):
    os.makedirs(root+name, exist_ok=True)
    out_dict ={}
    for key in path_dict:
        for item in path_dict[key]:
            out_dict[item] = label_dict[int(item.split('.')[0])]
            # Load image
            img = cv2.imread(root+'images/'+item)
            
            # Save image
            cv2.imwrite(root+name+'/'+item, img)    
    


    with open(root+name+'.json', 'w') as f:
        json.dump(out_dict, f)


save_dict_2_json(train_dict,data_dict,'/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/','train')
save_dict_2_json(val_dict,data_dict,'/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/','val')
save_dict_2_json(test_dict,data_dict,'/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/','test')
print('done')



# data_dict[‘top side’][‘length’][‘option1’] = short




