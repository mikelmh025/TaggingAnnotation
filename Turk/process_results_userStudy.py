import json
import data_utils
import os
import cv2
from tqdm import tqdm
import csv
import numpy as np
import copy
import collections


root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/match2/'
label_csv = root + 'matching.csv'



attri_need = {
    '1':['Input.image_person1', 'Input.image_url1_1', 'Input.image_url1_2', 'Input.image_url1_3', 'Input.image_url1_4', 
    'Input.image_url1_5', 'Input.image_url1_6', 'Input.image_url1_7', 'Input.image_url1_8', 'Input.image_url1_9', 'Input.image_url1_10',], 

    '2':['Input.image_person2', 'Input.image_url2_1', 'Input.image_url2_2', 'Input.image_url2_3', 
            'Input.image_url2_4', 'Input.image_url2_5', 'Input.image_url2_6', 'Input.image_url2_7', 'Input.image_url2_8', 
            'Input.image_url2_9', 'Input.image_url2_10'],

    '3':['Input.image_person3', 'Input.image_url3_1', 'Input.image_url3_2', 
            'Input.image_url3_3', 'Input.image_url3_4', 'Input.image_url3_5', 'Input.image_url3_6', 'Input.image_url3_7', 
            'Input.image_url3_8', 'Input.image_url3_9', 'Input.image_url3_10',] , 

    'out':['Answer.result']
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


# Loop throught each row, save to vote dict. i.e. vote_dict[method_name][case/input_name] = [vote1, vote2, vote3]
vote_dict = {}
for row_idx, row in enumerate(rows):
    if row_idx==0: continue

    for case in attri_need_idx:
        if case == 'out': continue
        
        input_human_ = row[attri_need_idx[case][0]]
        input_name = input_human_.split('/')[-1].split('.')[0]
        input_options_ = row[attri_need_idx[case][1]:attri_need_idx[case][-1]+1]
        method_name = input_options_[0].split('/')[-3:-1]
        method_name = '_'.join(method_name)

        if method_name not in vote_dict:
            vote_dict[method_name] = {}

        # Process annotated results
        output_result_ = row[attri_need_idx['out'][0]]
        try:
            output_result_ = string2dict(output_result_)['group'+case]
        except:
            print('Error:', output_result_)
            continue
        output_result_ = output_result_.split('/')[-1].split('.')[0]
        output_name, output_target = output_result_.split('_')[0], output_result_.split('_')[1]


        input_target = None
        target_list  = []
        for option in input_options_:
            if option == 'zzzz': continue
            option_name = option.split('/')[-1].split('.')[0]
            option_image, option_target = option_name.split('_')[0], option_name.split('_')[1]
            if input_name == option_image: 
                input_target = option_target
                # break
            target_list.append(option_target)
        assert input_target is not None, 'input_target is None'
        if input_name not in vote_dict[method_name]:
            vote_dict[method_name][input_name] = {
                'input_target':input_target,
                'output_target':[output_target],
                'target_list':target_list,
            }
        else:
            vote_dict[method_name][input_name]['output_target'].append(output_target)
            # vote_dict[method_name][input_name]['target_list'] = target_list        

correct_dict = {}
debug_list = {}
# debug_method = 'test_direct_bd_test_mapped1'
# debug_method = 'test_tag_bd_aggre'
debug_method = 'test_tag_pred_top1'

for method_name in vote_dict:
    if method_name not in correct_dict:
        correct_dict[method_name] = [0,0]
    for input_name in vote_dict[method_name]:
        
        input_target = vote_dict[method_name][input_name]['input_target']
        votes = vote_dict[method_name][input_name]['output_target']    
        votes_counter = collections.Counter(votes)
        aggre_works = max(votes_counter.values()) > 1
        if not aggre_works: continue
        output_target = max(set(votes), key=votes.count)
        target_list = vote_dict[method_name][input_name]['target_list']


        if output_target == input_target:
            correct_dict[method_name] = [correct_dict[method_name][0]+1, correct_dict[method_name][1]+1]
        else:
            correct_dict[method_name] = [correct_dict[method_name][0], correct_dict[method_name][1]+1]
            if method_name == debug_method:
                debug_list[int(input_name)] = [target_list,[input_target],[output_target]] 
a=1
correct_dict = dict(sorted(correct_dict.items(), key=lambda item: item[0]))

for key in correct_dict:
    print(key, round(100*correct_dict[key][0]/correct_dict[key][1],2))
    if key == debug_method:
        assert len(debug_list) == correct_dict[key][1]-correct_dict[key][0]



import data_utils

human_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/test'
asset_root = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/images'
save_root  = root+'debug/'
os.makedirs(save_root, exist_ok=True)


from search_algo  import search_algorithm 
asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/' + 'asset/820_faceattribute_round2_asset_translate_soft.json'
human_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/' + 'FairFace2.0/' + 'all_results_soft' + '.json'
with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

with open(human_json_path, 'r') as f:
    human_data = json.load(f)
# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))
asset_data = sort_dict(asset_data)
human_data = sort_dict(human_data)
algo = search_algorithm()


dis_sum_list = []
# sort debug_list by key
debug_list = dict(sorted(debug_list.items(), key=lambda item: item[0]))
for key in debug_list:
    human_path = os.path.join(human_root, str(key)+'.jpg')
    corr_path  = os.path.join(asset_root, debug_list[key][1][0]+'.png')
    pred_path  = os.path.join(asset_root, debug_list[key][2][0]+'.png')
    
    other_options = debug_list[key][0]
    other_options.remove(debug_list[key][1][0])
    other_options.remove(debug_list[key][2][0])
    other_options = [os.path.join(asset_root, item+'.png') for item in other_options]

    dis_dict_corr, dis_sum_corr = algo.eval_distance(human_data[str(key)],asset_data[debug_list[key][1][0]+'.png'])
    dis_dict_corr2, dis_sum_corr2 = algo.eval_distance_specific(human_data[str(key)], asset_data[debug_list[key][1][0]+'.png'],['top_curly','side_curly'])
    dis_sum_corr += dis_sum_corr2*0.5

    dis_dict_pred, dis_sum_pred = algo.eval_distance(human_data[str(key)],asset_data[debug_list[key][2][0]+'.png'])
    dis_dict_pred2, dis_sum_pred2 = algo.eval_distance_specific(human_data[str(key)], asset_data[debug_list[key][2][0]+'.png'],['top_curly','side_curly'])
    dis_sum_pred += dis_sum_pred2*0.5

    # dis_dict, dis_sum = algo.eval_distance(asset_data[debug_list[key][1][0]+'.png'], asset_data[debug_list[key][2][0]+'.png'])
   

    if dis_sum_pred >=0 :
        dis_sum = round(dis_sum_pred,2)
        dis_sum_list.append(dis_sum)

        dist_detail_corr,dist_detail_pred = '',''
        for item in dis_dict_corr:
            if dis_dict_corr[item] != 0: dist_detail_corr += item+': '+str(round(dis_dict_corr[item],2))+' \n'
            if dis_dict_pred[item] != 0: dist_detail_pred += item+': '+str(round(dis_dict_pred[item],2))+' \n'

        matched_asset_paths = [human_path, corr_path, pred_path] + other_options
        # matched_titles = ['']*len(matched_asset_paths)
        matched_titles = ['in ','corr \n'+dist_detail_corr,'annotated \n'+dist_detail_pred] + ['']*len(other_options)
        im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles)

        save_path = os.path.join(save_root, str(key)+'.jpg')
        cv2.imwrite(save_path, im_concat)
        print(key, debug_list[key])

print('average distance:', np.mean(dis_sum_list))
medium = np.median(dis_sum_list)
print('median distance:', medium)