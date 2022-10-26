import data_utils
import os
import csv
import random
import cv2

root = '/Users/minghaoliu/Desktop/Data_HITL_navi/'

url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/Data_HITL_navi/'

human_dir = 'test'
mode = 'matching'
# mode = 'subjective'
mode = ['matching','subjective']

match_options = 4
match_counter_dist_thredhold = 5

csv_save_dir = 'turk_csv/'
if not os.path.exists(os.path.join(root,csv_save_dir)):
    os.makedirs(os.path.join(root,csv_save_dir), exist_ok=True)

# Note: Other system: show three images, need to collect data
# 'other_system':['other_system/cartoonset100k/save_dir/0110','other_system/metahuman/men_20'],


# target_dir_dict = {
#     'test_direct_bd':['test_mapped1','test_mapped2','test_mapped3'],
#     'test_direct_pred':['pred'],
#     'test_direct_turk':['Turk 1','Turk 2','Turk 3'],
#     'test_other1_pred':[],
#     'test_other2_pred':[],
#     'test_other3_pred':[],
#     'test_tag_bd':['1','2','3','aggre'],
#     'test_tag_pred':['top1','top2','top3','top4','top5'],
#     'test_tag_turk':['_match1','_match2','_match3','_match_aggre']
# }

target_dir_dict = {
    'test_direct_bd':['test_mapped1'],
    'test_direct_pred':[],
    'test_direct_turk':['Turk 1'],
    'test_other1_pred':[],
    'test_other2_pred':[],
    'test_other3_pred':[],
    'test_tag_bd':['1','aggre'],
    'test_tag_pred':[],
    'test_tag_turk':['_match1']
}

##### Search algorithm #####
import data_utils
import json
from search_algo  import search_algorithm 
human_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/test'
asset_root = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/images'
asset_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/820_faceattribute_round2_asset_translate_soft.json'
human_json_path = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/all_results_soft.json'
with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)
with open(human_json_path, 'r') as f:
    human_data = json.load(f)
# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))
asset_data,human_data = sort_dict(asset_data), sort_dict(human_data)
algo = search_algorithm()
##### Search algorithm #####


human_image_paths = data_utils.make_im_set(os.path.join(root, human_dir))
# for human_image_path in human_image_paths:
#     human_image_name = os.path.basename(human_image_path)

if 'subjective' in mode:
    combined_csv_path = os.path.join(root, csv_save_dir, 'combined_subjective.csv')
    with open(combined_csv_path, 'w') as f:
        writer = csv.writer(f)
        header  = ['image_person1','image_cartoon1']
        header += ['image_person2','image_cartoon2']
        header += ['image_person3','image_cartoon3']
        writer.writerow(header)

    for target_dir in target_dir_dict:  # IE. test_direct_bd
        for target_subdir in target_dir_dict[target_dir]:  # IE. test_mapped1
            target_image_paths = data_utils.make_im_set(os.path.join(root, target_dir, target_subdir))

            # Create csv file
            csv_path = os.path.join(root, csv_save_dir, target_dir+ '_' + target_subdir + '_subjective.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

            buffer_list = []
            for target_image_path in target_image_paths:
                human_name = target_image_path.split('/')[-1].split('_')[0]
                target_name = target_image_path.split('/')[-1].split('_')[1].replace('.jpg','')

                human_path_ = os.path.join(root, human_dir, human_name+'.jpg')
                assert os.path.exists(human_path_), human_path_
                assert os.path.exists(target_image_path), target_image_path
                human_url_ = human_path_.replace(root, url_root)
                target_url_ = target_image_path.replace(root, url_root)

                buffer_list+= [human_url_, target_url_]
                if len(buffer_list) == len(header):
                    # Write to individual CSV
                    with open(csv_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(buffer_list)
                    # Write to combined CSV
                    with open(combined_csv_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(buffer_list)
                    buffer_list = []
            # break

if 'matching' in mode:
    template_dict = {}
    combined_csv_path = os.path.join(root, csv_save_dir, 'combined_matching.csv')
    with open(combined_csv_path, 'w') as f:
        writer = csv.writer(f)
        
        header = ['image_person1','image_url1_1','image_url1_2','image_url1_3','image_url1_4','image_url1_5','image_url1_6','image_url1_7','image_url1_8','image_url1_9','image_url1_10']
        header += ['image_person2','image_url2_1','image_url2_2','image_url2_3','image_url2_4','image_url2_5','image_url2_6','image_url2_7','image_url2_8','image_url2_9','image_url2_10']
        header += ['image_person3','image_url3_1','image_url3_2','image_url3_3','image_url3_4','image_url3_5','image_url3_6','image_url3_7','image_url3_8','image_url3_9','image_url3_10']
        writer.writerow(header)

    for target_dir in target_dir_dict:  # IE. test_direct_bd
        for target_subdir in target_dir_dict[target_dir]:  # IE. test_mapped1
            target_image_paths = data_utils.make_im_set(os.path.join(root, target_dir, target_subdir))
            target_image_paths = sorted(target_image_paths)
            target_image_dict = {os.path.basename(item).split('.')[0].split('_')[0]:item for item in target_image_paths}

            # Create csv file
            # csv_path = os.path.join(root, target_dir, target_subdir + '_matching.csv')
            csv_path = os.path.join(root, csv_save_dir, target_dir+ '_' + target_subdir + '_matching.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
            # Process each human image / Case
            buffer_list = []
            for target_image_path in target_image_paths:
                human_name = target_image_path.split('/')[-1].split('_')[0]
                target_name = target_image_path.split('/')[-1].split('_')[1].replace('.jpg','')

                human_path_ = os.path.join(root, human_dir, human_name+'.jpg')
                assert os.path.exists(human_path_), human_path_
                assert os.path.exists(target_image_path), target_image_path
                human_url_ = human_path_.replace(root, url_root)
                target_url_ = target_image_path.replace(root, url_root)

                ''' If no template, create one by randonly selecting counter samples '''
                if human_name not in template_dict:
                    # Get other images
                    other_image_paths = target_image_paths.copy()
                    other_image_paths.remove(target_image_path)
                    other_image_paths_selected = []

                    # Assert counte sample must have distance larger than threshold (5), Hence less aimbiguity during the annotation process
                    for other_image_path in other_image_paths:
                        other_image_name = os.path.basename(other_image_path).split('.')[0].split('_')[0]
                        other_image_target = os.path.basename(other_image_path).split('.')[0].split('_')[1]
                        asset_ = asset_data[other_image_target+'.png']
                        human_ = human_data[human_name]
                        dis_dict_corr, dis_sum_corr = algo.eval_distance(human_, asset_)

                        if dis_sum_corr >= dis_sum_corr:
                            other_image_paths_selected.append(other_image_path)

                    random.shuffle(other_image_paths_selected)

                    other_image_paths_selected = other_image_paths_selected[:match_options-1]

                    all_options = [target_url_] + [other_image_path.replace(root, url_root) for other_image_path in other_image_paths_selected]
                    random.shuffle(all_options)
                    all_options = all_options + ['zzzz']* (10-len(all_options))
                    all_options = [human_url_] + all_options
                    template_dict[human_name] = all_options
                else:
                    all_options = template_dict[human_name]
                    for i in range(len(all_options)):
                        if all_options[i] == 'zzzz':
                            continue
                        elif os.path.join(url_root, human_dir)+'/' in all_options[i]:
                            continue
                        else:
                            template_counter_case_name = all_options[i].split('/')[-1].split('_')[0]
                            counter_case_path_ = target_image_dict[template_counter_case_name]
                            assert os.path.exists(counter_case_path_), counter_case_path_
                            counter_case_url_ = counter_case_path_.replace(root, url_root)
                            all_options[i] = counter_case_url_

                buffer_list += all_options
                if len(buffer_list) == len(header):
                    # Write to individual CSV
                    with open(csv_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(buffer_list)
                    # Write to combined CSV
                    with open(combined_csv_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(buffer_list)
                    

                    #Debug: save results to image
                    save_dir = csv_path.replace('.csv', '')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    name = buffer_list[0].split('/')[-1].split('_')[0]
                    save_path = os.path.join(save_dir, name+'.jpg')
                    while 'zzzz' in buffer_list: buffer_list.remove('zzzz')
                        
                    matched_asset_paths = [item.replace(url_root, root) for item in buffer_list]
                    matched_titles = ['']*len(matched_asset_paths)
                    im_concat = data_utils.concat_list_image(matched_asset_paths,matched_titles)
                    cv2.imwrite(save_path, im_concat)
                    #Debug: save results to image

                    buffer_list = []

        # break
