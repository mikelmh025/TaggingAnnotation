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

# human_names = ['all_results_individual_1','all_results_individual_2','all_results_individual_3','all_results_soft']
human_names = ['all_results_soft']

# Bitmoji
asset_json_path = root + 'asset/820_faceattribute_round2_asset_translate_soft.json'
asset_dir       = root + 'asset/images'

# Google cartoon
# asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/google_cartoon_results_soft_labels.json'
# asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/google_cartoon_single'

# Metahuman (Female/male same tags)
asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/metahuman_results_soft_labels.json'
asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/metahuman_female'
# asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/metahuman_male'

# NovelAI male (Female/male different tags)
# asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/NovelAI_male_results_soft_labels.json'
# asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/NovelAI_male'

# NovelAI female (Female/male different tags)
# asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/NovelAI_female_results_soft_labels.json'
# asset_dir       = '/Users/minghaoliu/Desktop/HITL_navi/data/other_system/NovelAI_female'


image_pre_fix = ''
human_image_dir = root + label_image_dir + image_subset
human_image_paths = data_utils.make_im_set(human_image_dir)

show_top = 2#200
# search_top = 5
# assert search_top>=show_top

# Manual
# bitmoji_auto_dir = root + 'asset/3_bitmoji_auto'
# bitmoji_auto_paths = data_utils.make_im_set(bitmoji_auto_dir)
# bitmoji_auto_paths = []
# bitmoji_manual_dir = root + 'asset/4_bitmoji_manual'
# bitmoji_manual_paths = data_utils.make_im_set(bitmoji_manual_dir)
# bitmoji_manual_paths =[]
# invalid_path = root + 'asset/invalid.png'
# invalid_path = '/Users/bytedance/Desktop/data/annotations/811_fairV3_430_nto1/invalid.png'



with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))


asset_data = sort_dict(asset_data)

algo = search_algorithm()



# use tqdm to show progress bar in human_names loop


for human_name in tqdm(human_names):
    save_dir = root + label_image_dir + human_name+'_'+image_subset+'_mapped'

    human_json_path = root + label_image_dir + human_name+'.json'#'820_faceattribute_round2_v3_translate_relax.json' 
    with open(human_json_path, 'r') as f:
        human_data = json.load(f)
    human_data = sort_dict(human_data)
    coutner = 0
    for human_image_path in tqdm(human_image_paths):
        human_key  = human_image_path.split('/')[-1].split('.')[0]
        coutner += 1
        # if '12999' not in human_key: continue
        matched_asset_paths, matched_titles, image_name, matched_asset_dis_reports =  \
            algo.get_one_matched(human_data[human_key],asset_data,human_key,
                human_image_dir=human_image_dir,asset_dir=asset_dir,
                attr_care = ['top_curly','side_curly'],search_top=15)
        matched_asset_paths = matched_asset_paths[:show_top+1]
        if save_match_img:
            # '''Concatenated version'''
            im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles=matched_titles)
            cont_save_dir = str(save_dir)+'_concatenate'
            os.makedirs(cont_save_dir, exist_ok=True)
            cv2.imwrite(str(cont_save_dir+'/'+image_name+'.jpg'), im_concat)

            '''Individual save'''
            match_save_dir = str(save_dir)+'_match'
            os.makedirs(match_save_dir, exist_ok=True)
            paired_image = cv2.imread(matched_asset_paths[1])
            paired_image_name = matched_asset_paths[1].split('/')[-1].split('.')[0]
            cv2.imwrite(str(match_save_dir+'/'+image_name+'_'+paired_image_name+'.jpg'), paired_image)

            for top_i in range(1,show_top+1):
                top_save_dir = str(save_dir)+'_top'+str(top_i)
                os.makedirs(top_save_dir, exist_ok=True)
                top_image = cv2.imread(matched_asset_paths[top_i])
                top_image_name = matched_asset_paths[top_i].split('/')[-1].split('.')[0]
                cv2.imwrite(str(top_save_dir+'/'+image_name+'_'+top_image_name+'.jpg'), top_image)

        # if coutner >= 100: break
        
            

# def get_one_matched(human_data,asset_data, human_key):
#     attr_care = ['top_curly','side_curly']
#     image_name = human_key.split('.')[0]

#     huamn_path = human_image_dir+'/'+image_pre_fix+human_key
#     if os.path.exists(huamn_path+'.png'):
#         huamn_path = huamn_path+'.png'
#     elif os.path.exists(huamn_path+'.jpg'):
#         huamn_path = huamn_path+'.jpg'
#     else:
#         huamn_path = huamn_path

#     # Inital round of search    
#     dis_scores1, dis_reports1 = algo.search_all_assets(human_data[human_key],asset_data)

#     # Trim asset pool based on round one search
#     round2_dict = {key:value for idx, (key, value) in enumerate(dis_scores1.items()) if idx < search_top}
#     asset_data_trimed = {key:value for key, value in asset_data.items() if key in round2_dict.keys()}
        
#     # Second round of search
#     dis_scores2, dis_reports2 = algo.search_all_assets(human_data[human_key],asset_data_trimed, attr_care=attr_care)


#     out_dis_scores, out_dis_reports = dis_scores2, dis_reports2

#     matched_asset_paths = [os.path.join(asset_dir, key) for key in out_dis_scores.keys()]
#     matched_asset_paths = matched_asset_paths[:show_top]


#     out_paths, out_titles = [huamn_path], ['human']
#     out_paths += matched_asset_paths
#     out_titles += [str(round(value,2)) for value in out_dis_scores.values()]

#     out_titles = ['']
#     for case in out_dis_scores:
        
#         r1_report = [attr + ' ' + str(round(dis_reports1[case][attr],2)) + ' \n' for attr in dis_reports1[case] if dis_reports1[case][attr] != 0 and attr not in attr_care] 
#         r1_report = ''.join(r1_report)
#         r2_report = [attr + ' ' + str(round(dis_reports2[case][attr],2)) + ' \n' for attr in dis_reports2[case] if dis_reports2[case][attr] != 0]
#         r2_report = ''.join(r2_report)

#         title_  = 'R1: \n' + r1_report + 'R2: \n' + r2_report
#         out_titles.append(title_)


#     return out_paths, out_titles, image_name, out_dis_reports

#     dis_list = []
#     result_score_dict = {}
#     for asset_key in asset_data:
        
#         dis_dict, dis_sum = algo.eval_distance(human_data[human_key],asset_data[asset_key])
#         dis_list.append((dis_sum, asset_key,dis_dict,human_data[human_key],asset_data[asset_key]))

#         if dis_sum not in result_score_dict :result_score_dict[dis_sum] = []
#         result_score_dict[dis_sum].append((asset_key,dis_dict,human_data[human_key],asset_data[asset_key]))
#     dis_list.sort(key=lambda x:x[0]) # Sort by distance
#     result_score_dict = sort_dict(result_score_dict)




#     huamn_path = human_image_dir+'/'+image_pre_fix+human_key
#     if os.path.exists(huamn_path+'.png'):
#         huamn_path = huamn_path+'.png'
#     elif os.path.exists(huamn_path+'.jpg'):
#         huamn_path = huamn_path+'.jpg'
#     else:
#         huamn_path = huamn_path


#     matched_asset_paths, matched_titles = [huamn_path], ['input']

#     # Show top 3 
#     # print(matched_asset_paths)
#     # out_report = [(item[0:2]) for item in dis_list[:show_top] ]
#     # dist_report = [(item[2]) for item in dis_list[:show_top] ]
#     # human_label = [(item[3]) for item in dis_list[:show_top] ]
#     # asset_label = [(item[4]) for item in dis_list[:show_top] ]
#     # matched_asset_paths += [asset_dir+'/'+image_pre_fix+item[1] for item in dis_list[:show_top]]
#     # matched_titles       += [str(round(item[0],2)) for item in dis_list[:show_top]]

#     # Each level show 2
#     for idx, level in enumerate(result_score_dict):
#         matched_asset_paths  += [asset_dir+'/'+image_pre_fix+item[0] for item in result_score_dict[float(level)][:10]]
#         matched_titles       += [str(round(item[1]['total'],2)) for item in result_score_dict[float(level)][:10]]
#         if len(matched_asset_paths) >= show_top+1: break
#     matched_asset_paths = matched_asset_paths[:search_top+1]

#     # Tune the second search space
#     # Round 2 Search
#     dis_list2 = []
#     result_score_dict2 = {}
#     for matched_asset_path in matched_asset_paths[1:]:
#         asset_key = matched_asset_path.split('/')[-1]
#         dis_dict2, dis_sum2 = algo.eval_distance_specific(human_data[human_key],asset_data[asset_key], attr_care=['top_curly','side_curly'])
#         dis_list2.append((dis_sum2, asset_key,dis_dict2,human_data[human_key],asset_data[asset_key]))

#         if dis_sum2 not in result_score_dict2 :result_score_dict2[dis_sum2] = []
#         result_score_dict2[dis_sum2].append((asset_key,dis_dict2,human_data[human_key],asset_data[asset_key]))
#         # if dis_sum2 == 0 : break
#     dis_list2.sort(key=lambda x:x[0]) # Sort by distance
#     result_score_dict2 = sort_dict(result_score_dict2)

#     matched_asset_paths2, matched_titles2 = [huamn_path], ['input']

#     for idx, level in enumerate(result_score_dict2):
#         matched_asset_paths2  += [asset_dir+'/'+image_pre_fix+item[0] for item in result_score_dict2[float(level)][:10]]
#         matched_titles2       += [str(round(item[1]['total'],2)) for item in result_score_dict2[float(level)][:10]]
#         if len(matched_asset_paths2) >= show_top+1: break
#     matched_asset_paths, matched_titles = matched_asset_paths2, matched_titles2
    

    

#     # Show bitmoji manual/auto result
#     # if bitmoji_auto_dir+'/'+image_name+'.jpg' in bitmoji_auto_paths:
#     #     cur_bitmoji_auto_path = bitmoji_auto_dir+'/'+image_name+'.jpg'
#     #     cur_bimojit_manual_path = bitmoji_manual_dir+'/'+image_name.split('_')[0]+'.jpg'
#     # else:
#     #     cur_bitmoji_auto_path = invalid_path
#     #     cur_bimojit_manual_path = invalid_path
#     # matched_asset_paths.append(cur_bitmoji_auto_path)
#     # matched_titles.append("Auto")
#     # matched_asset_paths.append(cur_bimojit_manual_path)
#     # matched_titles.append("Manual")


#     # print(human_key, out_report)
#     matched_asset_paths = matched_asset_paths[:show_top+1]
#     matched_asset_paths = [ data_utils.fix_image_subscript(path.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','').replace('.png','.jpg')) for path in matched_asset_paths]
    

#     # # short mapped to long
#     # if '21658' in matched_asset_paths[0]:
#     #     a=1
#     # if '23617' in matched_asset_paths[0]:
#     #     a=1
    
#     return matched_asset_paths, image_name