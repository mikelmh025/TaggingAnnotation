import json
import data_utils
import os
import cv2
from search_algo  import search_algorithm 
from tqdm import tqdm
import csv
from itertools import groupby


save_match_img = False

root = '/Users/minghaoliu/Desktop/HITL_navi/data/'
label_image_dir = 'FairFace2.0/' #'v3/'#
# label_image_dir = 'FairFace2.0/' #'v3/'#
human_name = 'direct_bd'

image_subset = 'test'

asset_json_path = root + 'asset/820_faceattribute_round2_asset_translate_soft.json'
asset_dir       = root + 'asset/images'
image_pre_fix = ''
human_image_dir = root + label_image_dir + image_subset
human_image_paths = data_utils.make_im_set(human_image_dir)

annotation_rounds = 3


with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))


asset_data = sort_dict(asset_data)

save_dir = root + label_image_dir + human_name+'/'+image_subset+'_mapped'
# human_json_path = root + label_image_dir + human_name+'.json'
human_csv_path = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/direct_bd/annotation_translated.csv'

# Load csv to dict
human_data = {}
with open(human_csv_path, 'r') as f:
    reader = csv.reader(f)
    header = []
    for row in reader:
        if header == []: 
            header = row
            # task_id = header.index('Task ID')
            url_id  = header.index('url')
            Time_id1 = header.index('Time1')
            Selected1_id = header.index('Selected1')
            Time_id2 = header.index('Time2')
            Selected2_id = header.index('Selected2')
            Time_id3 = header.index('Time3')
            Selected3_id = header.index('Selected3')
            continue
        human_data[row[url_id]] = {} if row[url_id] not in human_data else human_data[row[url_id]]
        # human_data[row[url_id]]['Task ID'] = row[task_id]
        human_data[row[url_id]]['Time1'] = row[Time_id1]
        human_data[row[url_id]]['Time2'] = row[Time_id2]
        human_data[row[url_id]]['Time3'] = row[Time_id3]

        # string to dict row[Selected1_id]

        
        human_data[row[url_id]]['Selected1'] = json.loads(row[Selected1_id])['url'].replace('https://minghaouserstudy.s3.amazonaws.com/HITL_navi/bitmoji_asset/', '')
        human_data[row[url_id]]['Selected2'] = json.loads(row[Selected2_id])['url'].replace('https://minghaouserstudy.s3.amazonaws.com/HITL_navi/bitmoji_asset/', '')
        human_data[row[url_id]]['Selected3'] = json.loads(row[Selected3_id])['url'].replace('https://minghaouserstudy.s3.amazonaws.com/HITL_navi/bitmoji_asset/', '')



a=1

time_list = []
match_template = [0]*(annotation_rounds)

for case in human_data:

    cur_vot = []
    for i in range(1,annotation_rounds+1):
        time_list.append(float(human_data[case]['Time'+str(i)]))
        cur_vot.append(human_data[case]['Selected'+str(i)])

        if save_match_img:  # Save Selected image
            selected_image_path = asset_dir + '/' + human_data[case]['Selected'+str(i)]
            selected_image = data_utils.read_img(selected_image_path)

            save_dir_ = save_dir+str(i)
            if not os.path.exists(save_dir):os.makedirs(save_dir_, exist_ok=True)
            save_path = save_dir_+ '/'+ case + '_' + human_data[case]['Selected'+str(i)].replace('.png', '.jpg')
            cv2.imwrite(save_path, selected_image)
    
    aggrement = max([len(list(group)) for key, group in groupby(sorted(cur_vot))])
    match_template[aggrement-1] += 1

print('Average time: ', sum(time_list)/len(time_list))

aggre_works = 100*sum(match_template[1:])/sum(match_template)
print("Match aggrement",match_template, "aggre_works",round(aggre_works,2))