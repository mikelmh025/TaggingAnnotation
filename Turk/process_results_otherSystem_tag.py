import json
import data_utils
import os
import cv2
from search_algo  import search_algorithm 
from tqdm import tqdm
import csv
import numpy as np
import copy
import sys


save_match_img = True

root = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/'
# label_csv = root + 'google_cartoon_results.csv'
# label_csv = root + 'metahuman_results.csv'
# label_csv = root + 'NovelAI_female_results.csv'
label_csv = root + 'NovelAI_male_results.csv'



save_dir = root
# image_subset = 'test'

attri_need = ['Input.image_url1','Answer.results']
# human_names = ['all_results_soft']
# top_k = 4
# asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/'+ 'asset/820_faceattribute_round2_asset_translate_soft.json'
# asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/' + 'asset/images'
# human_image_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/test'
# with open(asset_json_path, 'r') as f:
#     asset_data = json.load(f)

# # Sort dict by key,return a dict
# def sort_dict(data):
#     return dict(sorted(data.items(), key=lambda d:d[0]))

# asset_data = sort_dict(asset_data)
# algo = search_algorithm()


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

#TODO: add entropy analysis 

all_soft_labels = {}
all_time_dict   = {}

for case in data_dict:
    if case not in all_soft_labels:
        all_soft_labels[case] = {}
        all_time_dict[case]   = {}



    for worker_idx, worker_ in enumerate(data_dict[case]):


        soft_label = {}
        time_dict = {}
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
                time_dict[combined_key] = worker_[key][item]['time']
        if soft_label == {} and time_dict == {}:
            continue
        all_soft_labels[case][worker_idx] = soft_label
        all_time_dict[case][worker_idx] = time_dict

# sort dict by key
all_soft_labels = {k: v for k, v in sorted(all_soft_labels.items(), key=lambda item: item[0])}
all_time_dict   = {k: v for k, v in sorted(all_time_dict.items(), key=lambda item: item[0])}



def get_one_matched(human_data,asset_data, human_key,top_k=5):
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
    matched_scores = [-1]


    # Each level show 2
    for idx, level in enumerate(result_score_dict):
        matched_asset_paths  += [asset_dir+'/'+item[0] for item in result_score_dict[float(level)]]
        matched_scores       += [float(level) for item in result_score_dict[float(level)]]
        matched_titles       += [str(round(item[1]['total'],2)) for item in result_score_dict[float(level)]]
        if len(matched_asset_paths) >= top_k+1: break


    


    matched_asset_paths = [ data_utils.fix_image_subscript(path.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','').replace('.png','.jpg')) for path in matched_asset_paths]

    return matched_asset_paths, image_name, matched_scores


# Check if Turk collection is completed
for key in all_soft_labels:
    if len(all_soft_labels[key])!=3:
        print(key)


# Combined dict of votes in to one soft dict
def combnine_dict(data_dict):
    result = None

    for key in data_dict:
        if result ==None:
            result = copy.deepcopy(data_dict[key])
        else:
            for attr in data_dict[key]:
                for opt in data_dict[key][attr]:
                    if opt not in result[attr]:
                        result[attr][opt] = 1
                    else:
                        result[attr][opt] += 1
    return result



# matched_dict = {}
for i in range(4):
    if not save_match_img: 
        print("Skip saving match images")
        break 

    turker_soft_labels_ = {}
    for key in all_soft_labels:
        
        try:    
            turker_soft_labels_[key] = all_soft_labels[key][i]
        except:
            try:
                turker_soft_labels_[key] = combnine_dict(all_soft_labels[key])  # if out of index use aggregated result
            except:
                print(key, all_soft_labels[key])
                raise

    # matched_dict[i] = {}

    # for human_key in turker_soft_labels_:
    #     if len(turker_soft_labels_[human_key]) == 0:
    #         continue
    #     matched_asset_paths, image_name, matched_scores = get_one_matched(turker_soft_labels_,asset_data,human_key,top_k=top_k)
    #     matched_dict[i][image_name] = matched_asset_paths
    #     # # break
    #     # continue    

    #     '''Concatenated version'''
    #     matched_titles = ['']*len(matched_asset_paths)
    #     im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles)
    #     cont_save_dir = str(save_dir)+'_concatenate'+str(i+1)
    #     os.makedirs(cont_save_dir, exist_ok=True)
    #     cv2.imwrite(str(cont_save_dir+'/'+image_name+'.jpg'), im_concat)

    #     '''Individual save'''
    #     match_save_dir = str(save_dir)+'_match'+str(i+1)
    #     os.makedirs(match_save_dir, exist_ok=True)
    #     paired_image = cv2.imread(matched_asset_paths[1])
    #     paired_image_name = matched_asset_paths[1].split('/')[-1].split('.')[0]
    #     cv2.imwrite(str(match_save_dir+'/'+image_name+'_'+paired_image_name+'.jpg'), paired_image)
    # a=1
json_save_path = label_csv.replace('.csv','_soft_labels.json')
with open(json_save_path, 'w') as f:
    json.dump(turker_soft_labels_, f)

#TODO complete time/entropy analysis
print('TODO complete time/entropy analysis')
sys.exit()

# Time analysis
Time_template  = {}
for case in all_time_dict:
    for worker_id in all_time_dict[case]:
        for attr in all_time_dict[case][worker_id]:
            if attr not in Time_template:
                Time_template[attr] = []
            Time_template[attr].append(all_time_dict[case][worker_id][attr]/1000)

total_time = 0
for attr in Time_template:
    time = round(np.mean(Time_template[attr]),2)
    total_time += time
    print(attr, time)

print('Total time', total_time)

a= 1

# Conduct entropy analysis (Tags)
entropy_templates = {'overall':[0]*len(all_soft_labels[key])}
for key in all_soft_labels:
    combined_soft_labels = combnine_dict(all_soft_labels[key])  # if out of index use aggregated result
    for attr in combined_soft_labels:
        if attr not in entropy_templates:
            entropy_templates[attr] = [0]*len(all_soft_labels[key])
        
        # get  max value in combined_soft_labels[attr]
        max_value = 0
        for opt in combined_soft_labels[attr]:
            max_value = combined_soft_labels[attr][opt] if combined_soft_labels[attr][opt] > max_value else max_value
        entropy_templates[attr][max_value-1] += 1
        entropy_templates['overall'][max_value-1] += 1

for key in entropy_templates:
    aggre_works = 100*sum(entropy_templates[key][1:])/sum(entropy_templates[key])
    print("Match aggrement",entropy_templates[key], "aggre_works",round(aggre_works,2))
aggre_works = 100*sum(entropy_templates['overall'][1:])/sum(entropy_templates['overall'])
print("Overall Match aggrement",entropy_templates['overall'], "aggre_works",round(aggre_works,2))
a=1

# Conduct entropy analysis (Final results)
entropy_final_templates = [0]*3
for case in matched_dict[0]:
    cur_matched = []
    for i in range(3):
        cur_matched += matched_dict[i][case][1:]  # only aggregate top 1
    unique, counts = np.unique(cur_matched, return_counts=True)
    match_count      = np.asarray((unique, counts)).T.tolist()
    match_count.sort(key=lambda x: x[1], reverse=True)
    max_count        = int(match_count[0][1])

    entropy_final_templates[max_count-1] += 1
    a=1

aggre_works = 100*sum(entropy_final_templates[1:])/sum(entropy_final_templates)
print("Final Match aggrement",entropy_final_templates, "aggre_works",round(aggre_works,2))
a=1


    