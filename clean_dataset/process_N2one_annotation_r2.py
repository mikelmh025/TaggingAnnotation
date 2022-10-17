import json
import data_utils
import os
import cv2
from search_algo  import search_algorithm 

# root = '/Users/bytedance/Desktop/data/annotations/820_faceattribute_round2/'
root = '/Users/minghaoliu/Desktop/HITL_navi/data/'
label_image_dir = 'FairFace2.0/' #'v3/'#
# label_image_dir = 'FairFace2.0/' #'v3/'#


# human_json_path = root + label_image_dir + 'all_results_relax.json' 
human_json_path = root + label_image_dir + 'all_results_soft.json'#'820_faceattribute_round2_v3_translate_relax.json' 

asset_json_path = root + 'asset/820_faceattribute_round2_asset_translate_soft.json'
asset_dir       = root + 'asset/images'
image_pre_fix = ''
human_image_dir = root + label_image_dir + 'images'

save_dir = root + label_image_dir + 'mapped'
show_top = 10

# Manual
bitmoji_auto_dir = root + 'asset/3_bitmoji_auto'
# bitmoji_auto_paths = data_utils.make_im_set(bitmoji_auto_dir)
bitmoji_auto_paths = []
bitmoji_manual_dir = root + 'asset/4_bitmoji_manual'
# bitmoji_manual_paths = data_utils.make_im_set(bitmoji_manual_dir)
bitmoji_manual_paths =[]
invalid_path = root + 'asset/invalid.png'
# invalid_path = '/Users/bytedance/Desktop/data/annotations/811_fairV3_430_nto1/invalid.png'

with open(human_json_path, 'r') as f:
    human_data = json.load(f)

with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))

human_data = sort_dict(human_data)
asset_data = sort_dict(asset_data)

a=1
# distance_score_quality= {
#     'blur': 0, 
#     'head_occlusion': 0, 
#     'mutiperson': 0
#     }

# Consider non-linear disntace for each level
# Consider some random number subscript for each level
# distance_score_linear = {
#     'top_curly': 1, 
#     'top_length': 1.5, 
#     'side_curly': 2, 
#     'side_length': 2, 
#     'braid_count': 8, 
#     }

# distance_score_type = {
#     'top_direction': 3, 
#     'braid_tf': 10, 
#     'braid_type': 3, 
#     'braid_position': 2}

# top_direction_group1     = ['0-向下','1-斜下','2-横向','3-斜上','6-向上（头发立起来）','7-向后梳头（长发向后梳，大背头，马尾）']
# top_direction_group2     = ['4-中分','5-37分']
# top_direction_group1_int = [ data_utils.attr2int(attr) for attr in top_direction_group1]
# top_direction_group2_int = [ data_utils.attr2int(attr) for attr in top_direction_group2]


# TODO: MAP data from label to bitmoji asset

# def eval_distance(human,asset,check=False):
#     def type_distance(human,asset):
#         return int(len(data_utils.intersection_list(human,asset)) <1)

#     # Find the min pairs from two lists
#     def linear_distance(human,asset):
#         min_distance = 99999
#         for i in range(len(human)):
#             for j in range(len(asset)):
#                 min_distance = min(min_distance,abs(human[i]-asset[j]))
#         return min_distance

#     distance_dict = {'total':0}
#     for attr in human:
#         human_attris = [ data_utils.attr2int(attr) for attr in human[attr]]
#         asset_attris = [ data_utils.attr2int(attr) for attr in asset[attr]]

        
#         if attr in distance_score_quality:
#             distance_ = distance_score_quality[attr] * type_distance(human_attris,asset_attris)# Type distance: Type error when no intersection
#         elif attr in distance_score_linear:
#             mean_human, mean_asset = sum(human_attris)/len(human_attris), sum(asset_attris)/len(asset_attris)
#             distance_  = distance_score_linear[attr] * abs(mean_human - mean_asset) # Linear distance: absolute difference
#             # distance_  = distance_score_linear[attr] * (linear_distance(human_attris, asset_attris))
#         elif attr in distance_score_type:
#             if attr == 'top_direction':    
#                 human_attris_g1, human_attris_g2 = data_utils.intersection_list(human_attris,top_direction_group1_int), data_utils.intersection_list(human_attris,top_direction_group2_int)
#                 asset_attris_g1, asset_attris_g2 = data_utils.intersection_list(asset_attris,top_direction_group1_int), data_utils.intersection_list(asset_attris,top_direction_group2_int)
#                 distance1 = distance_score_type[attr] * type_distance(human_attris_g1,asset_attris_g1) # Type distance
#                 distance2 = distance_score_type[attr] * type_distance(human_attris_g2,asset_attris_g2) # Type distance
#                 distance_ = [distance1,distance2]
#             else:
#                 distance_ = distance_score_type[attr] * type_distance(human_attris,asset_attris) # Type distance: Type error when no intersection

#         distance_sum = distance_ if type(distance_) != list else sum(distance_)
#         # if distance_sum != 0:
#         #     print(attr, distance_)
#         distance_dict[attr] = distance_
#         distance_dict['total'] += distance_sum

#     return distance_dict, distance_dict['total']
algo = search_algorithm()

for human_key in human_data:
    image_name = human_key.split('.')[0]

    dis_list = []
    result_score_dict = {}
    for asset_key in asset_data:
        temp = True if '1333' in asset_key and '005'in human_key else False
        dis_dict, dis_sum = algo.eval_distance(human_data[human_key],asset_data[asset_key],check=temp)
        dis_list.append((dis_sum, asset_key,dis_dict,human_data[human_key],asset_data[asset_key]))

        if dis_sum not in result_score_dict :result_score_dict[dis_sum] = []
        result_score_dict[dis_sum].append((asset_key,dis_dict,human_data[human_key],asset_data[asset_key]))
    dis_list.sort(key=lambda x:x[0]) # Sort by distance
    result_score_dict = sort_dict(result_score_dict)


    
    matched_asset_paths, matched_titles = [human_image_dir+'/'+image_pre_fix+human_key], ['input']

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
    a=1

    matched_asset_paths = [ data_utils.fix_image_subscript(path.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','').replace('.png','.jpg')) for path in matched_asset_paths]

    im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles)
    cont_save_dir = str(save_dir)+'_concatenate'
    os.makedirs(cont_save_dir, exist_ok=True)
    cv2.imwrite(str(cont_save_dir+'/'+image_name+'.jpg'), im_concat)

    match_save_dir = str(save_dir)+'_match'
    os.makedirs(match_save_dir, exist_ok=True)
    paired_image = cv2.imread(matched_asset_paths[1])
    paired_image_name = matched_asset_paths[1].split('/')[-1].split('.')[0]
    cv2.imwrite(str(match_save_dir+'/'+image_name+'_'+paired_image_name+'.jpg'), paired_image)

        
            