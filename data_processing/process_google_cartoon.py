import data_utils
import csv
import os
import cv2
import random

root = '/Users/minghaoliu/Desktop/Data_HITL_navi/other_system/cartoonset100k/'
save_dir    = '/Users/minghaoliu/Desktop/Data_HITL_navi/other_system/cartoonset100k/save_dir'
img_paths,csv_paths = [] , []

# for i in range(10):
for i in range(10):
    dataset_dir = root + str(i) + '/'
    img_paths += data_utils.make_im_set(dataset_dir)
    csv_paths += data_utils.make_csv_set(dataset_dir)
# sort img_paths by name
img_paths.sort(key=lambda x:x.split('/')[-1].split('.')[0])
csv_paths.sort(key=lambda x:x.split('/')[-1].split('.')[0])


template = None

ignore_val = [
'eye_angle',
'eye_lashes',
'eye_lid',
'eyebrow_weight',
'facial_hair',
'eye_color',
'eyebrow_width',
'eye_eyebrow_distance',

'eyebrow_shape',
'eyebrow_thickness',
'face_color',
'hair_color',
'face_shape',

'eye_slant']


# 'chin_length',
# 'glasses',
# 'glasses_color',

def load_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def process_template(data):
    coded_val = ""
    key = ""
    for i in range(len(data)):
        if data[i][0]!='hair':
            if data[i][0] in ignore_val:
                continue
            coded_val += str(data[i][1]).replace(' ','')
        else:
            key = data[i][1].replace(' ','')

    return coded_val, key


# Note hair row idx is 9
# Note hair tyle has 111
hair_dict = {}

counter = 0

for image_path in img_paths:
    if counter >= 3: break

    template = load_csv(image_path.replace('png','csv'))
    if template[8][1].replace(' ','') != '14': continue

    coded_val, hair_key = process_template(template)
    if coded_val not in hair_dict: hair_dict[coded_val] = {}
    if hair_key not in hair_dict[coded_val]: hair_dict[coded_val][hair_key] = []
    hair_dict[coded_val][hair_key].append(image_path)

    # test = len(list(hair_dict[coded_val].keys()))>=111
    # if test:
    #     counter += 1
    #     print('test!')


satified_dict = {}
for encode_ in hair_dict:
    satified = len(list(hair_dict[encode_].keys())) >=111
    if satified:
        satified_dict[encode_] = hair_dict[encode_]
a=1

for encode_ in satified_dict:
    save_path = os.path.join(save_dir, encode_)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for hair_type in satified_dict[encode_]:
        

        source_paths = satified_dict[encode_][hair_type]
        # randmly select 3 images from a list source_paths
        random.shuffle(source_paths)
        source_paths = source_paths[:3]

        for idx, path in enumerate(source_paths):
            image_save_path = os.path.join(save_path, hair_type+'_'+str(idx)+'.png')
            img = cv2.imread(path)
            cv2.imwrite(image_save_path, img)
            # break
        # source_path = satified_dict[encode_][hair_type][0]
        # img = cv2.imread(source_path)
        # cv2.imwrite(image_save_path, img)
    


