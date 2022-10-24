import json
import data_utils
import os
import cv2
from search_algo  import search_algorithm 
from tqdm import tqdm
import csv

save_match_img = True

root = '/Users/minghaoliu/Desktop/Data_HITL_navi/test_tag_turk/'
label_csv = root + 'try2.csv'

save_dir = '/Users/minghaoliu/Desktop/Data_HITL_navi/test_tag_turk/'
image_subset = 'test'

attri_need = ['Input.image_url1','Answer.results']
# human_names = ['all_results_soft']
show_top = 3
asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/'+ 'asset/820_faceattribute_round2_asset_translate_soft.json'
asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/' + 'asset/images'
human_image_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/test'
with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))

asset_data = sort_dict(asset_data)
algo = search_algorithm()


# Load csv file
def load_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

rows = load_csv(label_csv)

data_dict = {}
header = rows[0]
idx_need = []
for i in range(len(header)):
    if header[i] in attri_need:
        idx_need.append(i)

def string2dict(string):
    string = string.replace('\'','')
    string = string.replace(' ','')
    string = string.replace('True','true')
    string = string.replace('False','false')
    string = string.replace('None','null')

    return json.loads(json.loads(string))

for row_idx, row in enumerate(rows):
    if row_idx==0:
        continue
    
    # data_dict[row_idx] = {}
    for i in range(len(idx_need)):

        if "https://minghaouserstudy.s3.amaz"in row[idx_need[i]]:
            input_img = row[idx_need[i]].split('/')[-1]
        else:
            try:
                processed_dict = string2dict(row[idx_need[i]])
            except:
                processed_dict = {}

    if input_img not in data_dict:
        data_dict[input_img]= []
    data_dict[input_img] += [processed_dict]
a=1

#TODO: add time analysis 

all_soft_labels = {}
for case in data_dict:
    if case not in all_soft_labels:
        all_soft_labels[case] = {}

    for worker_idx, worker_ in enumerate(data_dict[case]):


        soft_label = {}
        for key in worker_:
            region = key.lower()
            if 'Top' in key: 
                region = 'top'

            
            for item in worker_[key]:
                attr = item.lower()
                if 'curllevel' in attr:
                    attr = 'curly'
                if 'yes/no' in attr:
                    attr = 'tf'
                if 'style' in attr:
                    attr = 'type'

                combined_key = region + '_' + attr
                selected = str(int(worker_[key][item]['option'].replace('option','')) - 1)
                soft_label[combined_key] = {selected:1}
        
        all_soft_labels[case][worker_idx] = soft_label

# sort dict by key
all_soft_labels = {k: v for k, v in sorted(all_soft_labels.items(), key=lambda item: item[0])}
a=1
                


        # vote_dict = data_dict[case][worker_]
        # a=1

def get_one_matched(human_data,asset_data, human_key):
    image_name = human_key.split('.')[0]

    dis_list = []
    result_score_dict = {}
    for asset_key in asset_data:
        dis_dict, dis_sum = algo.eval_distance(human_data[human_key],asset_data[asset_key])
        dis_list.append((dis_sum, asset_key,dis_dict,human_data[human_key],asset_data[asset_key]))

        if dis_sum not in result_score_dict :result_score_dict[dis_sum] = []
        result_score_dict[dis_sum].append((asset_key,dis_dict,human_data[human_key],asset_data[asset_key]))
    dis_list.sort(key=lambda x:x[0]) # Sort by distance
    result_score_dict = sort_dict(result_score_dict)


    huamn_path = human_image_dir+'/'+human_key
    if os.path.exists(huamn_path+'.png'):
        huamn_path = huamn_path+'.png'
    elif os.path.exists(huamn_path+'.jpg'):
        huamn_path = huamn_path+'.jpg'
    else:
        huamn_path = huamn_path


    matched_asset_paths, matched_titles = [huamn_path], ['input']


    # Each level show 2
    for idx, level in enumerate(result_score_dict):
        matched_asset_paths  += [asset_dir+'/'+item[0] for item in result_score_dict[float(level)][:4]]
        matched_titles       += [str(round(item[1]['total'],2)) for item in result_score_dict[float(level)][:4]]
        if len(matched_asset_paths) >= show_top+1: break


    


    matched_asset_paths = [ data_utils.fix_image_subscript(path.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','').replace('.png','.jpg')) for path in matched_asset_paths]

    return matched_asset_paths, image_name

turker1_soft_labels = {}
for key in all_soft_labels:
    turker1_soft_labels[key] = all_soft_labels[key][0]

for human_key in turker1_soft_labels:
    if len(turker1_soft_labels[human_key]) == 0:
        continue
    matched_asset_paths, image_name = get_one_matched(turker1_soft_labels,asset_data,human_key)

    if save_match_img:
        '''Concatenated version'''
        matched_titles = ['']*len(matched_asset_paths)
        im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles)
        cont_save_dir = str(save_dir)+'_concatenate'
        os.makedirs(cont_save_dir, exist_ok=True)
        cv2.imwrite(str(cont_save_dir+'/'+image_name+'.jpg'), im_concat)

        '''Individual save'''
        match_save_dir = str(save_dir)+'_match'
        os.makedirs(match_save_dir, exist_ok=True)
        paired_image = cv2.imread(matched_asset_paths[1])
        paired_image_name = matched_asset_paths[1].split('/')[-1].split('.')[0]
        cv2.imwrite(str(match_save_dir+'/'+image_name+'_'+paired_image_name+'.jpg'), paired_image)
    a=1