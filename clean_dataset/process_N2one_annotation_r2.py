import json
import data_utils
import os
import cv2
from search_algo  import search_algorithm 
from tqdm import tqdm

save_match_img = True

root = '/Users/minghaoliu/Desktop/HITL_navi/data/'
label_image_dir = 'FairFace2.0/' #'v3/'#
# label_image_dir = 'FairFace2.0/' #'v3/'#

image_subset = 'test'

human_names = ['all_results_individual_1','all_results_individual_2','all_results_individual_3','all_results_soft']
# human_names = ['all_results_soft']

asset_json_path = root + 'asset/820_faceattribute_round2_asset_translate_soft.json'
asset_dir       = root + 'asset/images'
image_pre_fix = ''
human_image_dir = root + label_image_dir + image_subset
human_image_paths = data_utils.make_im_set(human_image_dir)

show_top = 3

# Manual
bitmoji_auto_dir = root + 'asset/3_bitmoji_auto'
# bitmoji_auto_paths = data_utils.make_im_set(bitmoji_auto_dir)
bitmoji_auto_paths = []
bitmoji_manual_dir = root + 'asset/4_bitmoji_manual'
# bitmoji_manual_paths = data_utils.make_im_set(bitmoji_manual_dir)
bitmoji_manual_paths =[]
invalid_path = root + 'asset/invalid.png'
# invalid_path = '/Users/bytedance/Desktop/data/annotations/811_fairV3_430_nto1/invalid.png'



with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))


asset_data = sort_dict(asset_data)

algo = search_algorithm()


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


    huamn_path = human_image_dir+'/'+image_pre_fix+human_key
    if os.path.exists(huamn_path+'.png'):
        huamn_path = huamn_path+'.png'
    elif os.path.exists(huamn_path+'.jpg'):
        huamn_path = huamn_path+'.jpg'
    else:
        huamn_path = huamn_path


    matched_asset_paths, matched_titles = [huamn_path], ['input']

    # Show top 3 
    # print(matched_asset_paths)
    # out_report = [(item[0:2]) for item in dis_list[:show_top] ]
    # dist_report = [(item[2]) for item in dis_list[:show_top] ]
    # human_label = [(item[3]) for item in dis_list[:show_top] ]
    # asset_label = [(item[4]) for item in dis_list[:show_top] ]
    # matched_asset_paths += [asset_dir+'/'+image_pre_fix+item[1] for item in dis_list[:show_top]]
    # matched_titles       += [str(round(item[0],2)) for item in dis_list[:show_top]]

    # Each level show 2
    for idx, level in enumerate(result_score_dict):
        matched_asset_paths  += [asset_dir+'/'+image_pre_fix+item[0] for item in result_score_dict[float(level)][:4]]
        matched_titles       += [str(round(item[1]['total'],2)) for item in result_score_dict[float(level)][:4]]
        if len(matched_asset_paths) >= show_top+1: break


    

    # Show bitmoji manual/auto result
    # if bitmoji_auto_dir+'/'+image_name+'.jpg' in bitmoji_auto_paths:
    #     cur_bitmoji_auto_path = bitmoji_auto_dir+'/'+image_name+'.jpg'
    #     cur_bimojit_manual_path = bitmoji_manual_dir+'/'+image_name.split('_')[0]+'.jpg'
    # else:
    #     cur_bitmoji_auto_path = invalid_path
    #     cur_bimojit_manual_path = invalid_path
    # matched_asset_paths.append(cur_bitmoji_auto_path)
    # matched_titles.append("Auto")
    # matched_asset_paths.append(cur_bimojit_manual_path)
    # matched_titles.append("Manual")


    # print(human_key, out_report)

    matched_asset_paths = [ data_utils.fix_image_subscript(path.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','').replace('.png','.jpg')) for path in matched_asset_paths]

    return matched_asset_paths, image_name

# use tqdm to show progress bar in human_names loop


for human_name in tqdm(human_names):
    save_dir = root + label_image_dir + human_name+'_'+image_subset+'_mapped'

    human_json_path = root + label_image_dir + human_name+'.json'#'820_faceattribute_round2_v3_translate_relax.json' 
    with open(human_json_path, 'r') as f:
        human_data = json.load(f)
    human_data = sort_dict(human_data)
    coutner = 0
    for human_image_path in human_image_paths:
        human_key  = human_image_path.split('/')[-1].split('.')[0]
    # for human_key in tqdm(human_data):
        coutner += 1
        matched_asset_paths, image_name = get_one_matched(human_data,asset_data, human_key)
        
        if save_match_img:
            '''Concatenated version'''
            # im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles)
            # cont_save_dir = str(save_dir)+'_concatenate'
            # os.makedirs(cont_save_dir, exist_ok=True)
            # cv2.imwrite(str(cont_save_dir+'/'+image_name+'.jpg'), im_concat)

            '''Individual save'''
            match_save_dir = str(save_dir)+'_match'
            os.makedirs(match_save_dir, exist_ok=True)
            paired_image = cv2.imread(matched_asset_paths[1])
            paired_image_name = matched_asset_paths[1].split('/')[-1].split('.')[0]
            cv2.imwrite(str(match_save_dir+'/'+image_name+'_'+paired_image_name+'.jpg'), paired_image)

        
            