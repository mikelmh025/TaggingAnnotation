from email.mime import image
import data_utils
import csv
import pandas as pd 
import cv2
import os
from search_algo  import search_algorithm 
import json
# url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/test/'

human_img_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/test'
asset_img_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/images'

exp_root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/'
csv_path = exp_root+'init_direct_test204_results.csv'

save_image_root = csv_path.replace('.csv','_images/')
os.makedirs(save_image_root, exist_ok=True)

human_img_paths = data_utils.make_im_set(human_img_dir)
asset_img_paths = data_utils.make_im_set(asset_img_dir)

input_names = ['Input.image_url1']
output_names = ['Answer.submit_dict']


def merge_dict(dict1, dict2):
    for key in dict1:
        dict1[key] += dict2[key]
    
    return dict1

# Load csv file
result_dict = {}
df = pd.read_csv(csv_path)
for index, row in df.iterrows():
    input_urls    = [row[key] for key in input_names]
    output_list   = [eval(row[key]) for key in output_names]

    for num_task in range(len(input_names)):
        input_name = input_urls[num_task].split('/')[-1]
        output_label = output_list[num_task]
        for key in output_label:
            output_label[key] = [output_label[key]]
        
        result_dict[input_name] = output_label if input_name not in result_dict else merge_dict(result_dict[input_name], output_label)


# sort result_dict by key
result_dict = {k: result_dict[k] for k in sorted(result_dict)}
a=1    


human_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/all_results_soft.json'
human_sample_json = json.load(open(human_sample_json_path))
human_json_dict = {}
for key in human_sample_json:
    clean_prefix_key = key.replace('0825_fair_face_clean_','')
    clean_prefix_key = clean_prefix_key.replace('0906_fair_face_clean_','')
    human_json_dict[clean_prefix_key] = human_sample_json[key]

asset_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/asset/820_faceattribute_round2_asset_translate_soft.json'
asset_sample_json = json.load(open(asset_sample_json_path))

algo = search_algorithm()

def search_best_match(human_path, human_json, asset_json, algo,top_k=5):
    human_name = human_path.split('/')[-1]
    human_label = human_json[human_name.replace('.jpg','.png')]

    score_dict = {}
    for key in asset_json: _,score_dict[key] = algo.eval_distance(human_label, asset_json[key])

    # sort score_dict by value
    score_dict = {k: score_dict[k] for k in sorted(score_dict, key=score_dict.get, reverse=False)}

    # get top_k
    top_k_list = list(score_dict.keys())[:top_k]
    top_k_scores = list(score_dict.values())[:top_k]
    return top_k_list, top_k_scores
        

# TODO: Show image results
for case in result_dict:
    input_path = human_img_dir+'/'+case
    matched_paths = [asset_img_dir+'/'+path+'.png' for path in result_dict[case]['hair']]
    matched_tile  = ['Turk '+str(i+1) for i in range(len(matched_paths))]

    top_k_paths,top_k_scores   = search_best_match(input_path, human_json_dict, asset_sample_json, algo, top_k=3)
    top_k_paths                = [asset_img_dir+'/'+path for path in top_k_paths]
    top_k_title                = ['T: '+str(top_k_scores[i]) for i in range(len(top_k_scores))]

    turk_image_paths = [input_path] + matched_paths #+ [ asset_img_dir+'/'+path for path in top_k_match]
    # Read images
    # all_images = [data_utils.read_img(path) for path in all_image_paths]
    
    # image_row = data_utils.horizontal_cat(all_images,column=0)
    im_concat1 = data_utils.concat_list_image(turk_image_paths,['input']+matched_tile)
    im_concat2 = data_utils.concat_list_image([input_path]+top_k_paths, ['']+top_k_title)

    im_concat = data_utils.vertical_cat([im_concat1,im_concat2])
    cv2.imwrite(save_image_root+case,im_concat)


# TODO: show time distribution


# TODO: show disagreement level 