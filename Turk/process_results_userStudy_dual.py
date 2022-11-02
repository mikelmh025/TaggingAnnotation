# import json
# import data_utils
# import os
# import cv2
# from tqdm import tqdm
# import csv
# import numpy as np
# import copy
# import collections


# # old version now!!!
# print('ERROR old version now!!!')

# root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/dual_run1/'
# # label_csv = root + 'Batch_4912817_batch_results.csv'
# # label_csv = root + 'pred_dual.csv'
# label_csv = root + 'bd_dual.csv'
# # label_csv = root + 'turk_dual.csv'




# attri_need = {
#     '1':['Input.image_person1','Input.image_cartoonA1','Input.image_cartoonB1'],
#     '2':['Input.image_person2','Input.image_cartoonA2','Input.image_cartoonB2'],
#     '3':['Input.image_person3','Input.image_cartoonA3','Input.image_cartoonB3'],
#     'out': ['Answer.result']
# }

# data_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/'
# url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/Data_HITL_navi/'

# attri_need_idx = {}
# for item in attri_need:
#     attri_need_idx[item] = [0]*len(attri_need[item])

# # Load csv file
# def load_csv(csv_path):
#     with open(csv_path, 'r') as f:
#         reader = csv.reader(f)
#         data = list(reader)
#     return data

# rows = load_csv(label_csv)

# data_dict = {}
# header = rows[0]



# for idx, item in enumerate(header):
#     for key in attri_need:
#         if item in attri_need[key]:
#             # find the index of item in attri_need[key]
#             index = attri_need[key].index(item)
#             attri_need_idx[key][index] = idx



# def string2dict(string):
#     string = string.replace('\'','')
#     string = string.replace(' ','')
#     string = string.replace('True','true')
#     string = string.replace('False','false')
#     string = string.replace('None','null')

#     return json.loads(string)

# # debug_list = []
# # TODO: deal with 3 cases

# vote_dict ={}
# method1_fail_counter, method2_fail_counter = 0, 0
# both_failed_counter, both_succeed_counter = 0, 0
# for row_idx, row in enumerate(rows):
#     if row_idx==0:
#         continue

    
#     for case in attri_need_idx:
#         if case == 'out': continue
        
#         input_human_ = row[attri_need_idx[case][0]]
#         input_name = input_human_.split('/')[-1].split('.')[0]


#         input_options_ = row[attri_need_idx[case][1]:attri_need_idx[case][-1]+1]
#         method_name1 = input_options_[0].split('/')[-3:-1]
#         method_name1 = '_'.join(method_name1)
#         method_name2 = input_options_[1].split('/')[-3:-1]
#         method_name2 = '_'.join(method_name2)

#         # Process annotated results
#         output_result = row[attri_need_idx['out'][0]]
#         output_result_ = string2dict(output_result)['group'+case]

#         if input_human_ not in vote_dict:
#             vote_dict[input_human_] = {}
#             vote_dict[input_human_]['scores'] = [output_result_]
#             vote_dict[input_human_]['options'] = [input_options_]
#         else:
#             vote_dict[input_human_]['scores'] += [output_result_]
#             vote_dict[input_human_]['options'] += [input_options_]

# correct_dict = {}      
# for case in vote_dict:
#     save_image = False
#     input_human_ = case
#     input_name = case.split('/')[-1].split('.')[0]
    

#     method_name1 = vote_dict[input_human_]['options'][0][0].split('/')[-3:-1]
#     method_name1 = '_'.join(method_name1)
#     method_name2 = vote_dict[input_human_]['options'][0][1].split('/')[-3:-1]
#     method_name2 = '_'.join(method_name2)

#     if method_name1 not in correct_dict:
#         correct_dict[method_name1] = [0,0]
#     if method_name2 not in correct_dict:
#         correct_dict[method_name2] = [0,0]

#     output_results = vote_dict[case]['scores']
#     while '0' in output_results:
#         output_results.remove('0')
#         correct_dict[method_name1] = [correct_dict[method_name1][0], correct_dict[method_name1][1]+1]
#         correct_dict[method_name2] = [correct_dict[method_name2][0], correct_dict[method_name2][1]+1]
#         both_failed_counter += 1
#     while '3' in output_results:
#         output_results.remove('3')
#         correct_dict[method_name1] = [correct_dict[method_name1][0]+1, correct_dict[method_name1][1]+1]
#         correct_dict[method_name2] = [correct_dict[method_name2][0]+1, correct_dict[method_name2][1]+1]
#         both_succeed_counter += 1
#     # count the number of 1 and 2
#     counter = collections.Counter(output_results)

#     diverge_votes = counter['1'] + counter['2']
#     if counter['1'] > counter['2']:
#         correct_dict[method_name1] = [correct_dict[method_name1][0]              , correct_dict[method_name1][1]+diverge_votes]
#         correct_dict[method_name2] = [correct_dict[method_name2][0]+diverge_votes, correct_dict[method_name2][1]+diverge_votes]
#         save_image = True
#         method1_fail_counter += diverge_votes
#     elif counter['1'] < counter['2']:
#         correct_dict[method_name1] = [correct_dict[method_name1][0]+diverge_votes, correct_dict[method_name1][1]+diverge_votes]
#         correct_dict[method_name2] = [correct_dict[method_name2][0]              , correct_dict[method_name2][1]+diverge_votes]
#         method2_fail_counter += diverge_votes

#     else:
#         # One bad vote for each method, do nothing.
#         a=1
    
#     # if save_image:
#     #     huamn_path = input_human_.replace(url_root, data_root)
#     #     concat_paths = [huamn_path]+[item.replace(url_root, data_root) for item in vote_dict[input_human_]['options'][0]]
#     #     concat_titles = ['human']+[method_name1, method_name2]
#     #     im_concat = data_utils.concat_list_image(concat_paths,concat_titles)
#     #     save_dir = label_csv.replace('.csv', '')
#     #     if not os.path.exists(save_dir):
#     #         os.makedirs(save_dir)
#     #     save_path = os.path.join(save_dir, input_name+'.png')
#     #     cv2.imwrite(save_path, im_concat)

# # sort correct_dict by key
# correct_dict = dict(sorted(correct_dict.items(), key=lambda item: item[0]))

# for key in correct_dict:
#     print(key, round(100*correct_dict[key][0]/correct_dict[key][1],2), 'countes: ', correct_dict[key][0], correct_dict[key][1])

# # debug_list.sort()
# # print(debug_list)

# print('method1_fail_counter: ', method1_fail_counter)
# print('method2_fail_counter: ', method2_fail_counter)
# print('both_failed_counter: ', both_failed_counter)
# print('both_succeed_counter: ', both_succeed_counter)

import json
import data_utils
import os
import cv2
from tqdm import tqdm
import csv
import numpy as np
import copy
import collections

# a = ['15924', '1771', '2055', '28669', '30378', '33176', '36401', '37280', '38171', '40465', '44945', '50802', '51969', '53861', '57282', '58684', '59928', '63055', '65929', '675', '68687', '71706', '72774', '77660', '77749', '80633', '81603', '81751']
# b = ['15924', '2055', '28669', '30378', '33176', '36401', '38171', '40465', '44945', '50802', '51969', '53861', '57282', '59928', '63055', '65929', '675', '68687', '71706', '72774', '77660', '77749', '80633']
# #intersection
# print('both in verion 3.0 and 3.1')
# print(set(a).intersection(set(b)))
# # difference
# print(set(a).difference(set(b)))
# print(set(b).difference(set(a)))

root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/dual_run2/'
# label_csv = root + 'Batch_4912817_batch_results.csv'
# label_csv = root + 'pred_dual.csv'
label_csv = root + 'bd_dual.csv'
# label_csv = root + 'turk_dual.csv'




attri_need = {
    '1':['Input.image_person1','Input.image_cartoonA1','Input.image_cartoonB1'],
    '2':['Input.image_person2','Input.image_cartoonA2','Input.image_cartoonB2'],
    '3':['Input.image_person3','Input.image_cartoonA3','Input.image_cartoonB3'],
    'out': ['Answer.result']
}

data_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/'
url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/Data_HITL_navi/'

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

worker_id_idx = header.index('WorkerId')

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

# debug_list = []
# TODO: deal with 3 cases
worker_id_list = []

vote_dict ={}
for row_idx, row in enumerate(rows):
    if row_idx==0:
        continue
    worker_id_list += [row[worker_id_idx]]


    
    for case in attri_need_idx:
        if case == 'out': continue
        
        input_human_ = row[attri_need_idx[case][0]]
        input_name = input_human_.split('/')[-1].split('.')[0]


        input_options_ = row[attri_need_idx[case][1]:attri_need_idx[case][-1]+1]
        method_name1 = input_options_[0].split('/')[-3:-1]
        method_name1 = '_'.join(method_name1)
        method_name2 = input_options_[1].split('/')[-3:-1]
        method_name2 = '_'.join(method_name2)

        # Process annotated results
        output_result = row[attri_need_idx['out'][0]]
        output_result_ = string2dict(output_result)['group'+case]

        if input_human_ not in vote_dict:
            vote_dict[input_human_] = {}
            vote_dict[input_human_]['scores'] = [output_result_]
            vote_dict[input_human_]['input_human_'] = input_human_
            vote_dict[input_human_]['options'] = input_options_
        else:
            vote_dict[input_human_]['scores'] += [output_result_]
            # vote_dict[input_human_]['options'] += [input_options_]

# get counts of unique worker id
worker_id_count = collections.Counter(worker_id_list)
print("worker_id_count",worker_id_count)

correct_dict = {}    

# Count the number of votes
method1_fail_counter, method2_fail_counter = 0, 0
both_failed_counter, both_succeed_counter = 0, 0  

# Record by case: 
# Note that there are overlapping cases
method1_fail_dict, method2_fail_dict = {}, {}
both_failed_dict, both_succeed_dict = {}, {}

for case in vote_dict:
    save_image = False
    input_human_ = case
    input_name = case.split('/')[-1].split('.')[0]
    

    method_name1 = vote_dict[input_human_]['options'][0].split('/')[-3:-1]
    method_name1 = '_'.join(method_name1)
    method_name2 = vote_dict[input_human_]['options'][1].split('/')[-3:-1]
    method_name2 = '_'.join(method_name2)

    asset_name1 = vote_dict[input_human_]['options'][0].split('/')[-1].split('.')[0]
    asset_name2 = vote_dict[input_human_]['options'][1].split('/')[-1].split('.')[0]

    if method_name1 not in correct_dict:
        correct_dict[method_name1] = [0,0]
    if method_name2 not in correct_dict:
        correct_dict[method_name2] = [0,0]

    output_results = vote_dict[case]['scores'].copy()
    votes_counter = collections.Counter(output_results)
    aggre_works = max(votes_counter.values()) > 1
    
    # Here each vote represent a Bad case
    if aggre_works:
        # get key of max value from dict votes_counter
        max_key = max(votes_counter, key=votes_counter.get)
    else:
        # bad_score_ = [0,0]
        # for key in votes_counter:
        #     assert votes_counter[key] == 1
        #     if key =='0':
        #         bad_score_[0] += 1
        #         bad_score_[1] += 1
        #     elif key == '1':
        #         bad_score_[0] += 1
        #     elif key == '2':
        #         bad_score_[1] += 1
        #     elif key =='3':
        #         bad_score_ = bad_score_
        #     else:
        #         bad_score_ = bad_score_
        #         # assert False, 'Wrong key'
        # if bad_score_[0] > bad_score_[1]:
        #     max_key = '1'
        # elif bad_score_[0] < bad_score_[1]:
        #     max_key = '2'
        # else:
        #     if sum(bad_score_)/len(bad_score_) >= 1.5:
        #         max_key = '0'
        #     else:
        #         max_key = '3'

        good_score_ = [0,0]
        for key in votes_counter:
            assert votes_counter[key] == 1
            if key =='0':
                good_score_ = good_score_
            elif key == '1':
                good_score_[0] += 1
            elif key == '2':
                good_score_[1] += 1
            elif key =='3':
                good_score_[0] += 1
                good_score_[1] += 1
            else:
                good_score_ = good_score_
                # assert False, 'Wrong key'
        if good_score_[0] > good_score_[1]:
            max_key = '1'
        elif good_score_[0] < good_score_[1]:
            max_key = '2'
        else:
            if sum(good_score_)/len(good_score_) >= 1.5:
                max_key = '3'
            else:
                max_key = '0'

    # if asset_name1 == asset_name2 : # denoise when the two methods are the same
    #     max_key = '3'
    
    save_image = False
    if max_key == '0':
        correct_dict[method_name1] = [correct_dict[method_name1][0], correct_dict[method_name1][1]+1]
        correct_dict[method_name2] = [correct_dict[method_name2][0], correct_dict[method_name2][1]+1]
        both_failed_counter += 1
        both_failed_dict[input_name] = vote_dict[input_human_]
        # save_image = True
    elif max_key == '1': # Using good score isnteading of bad score
        # correct_dict[method_name1] = [correct_dict[method_name1][0]  , correct_dict[method_name1][1]+1]
        # correct_dict[method_name2] = [correct_dict[method_name2][0]+1, correct_dict[method_name2][1]+1]
        correct_dict[method_name1] = [correct_dict[method_name1][0]+1, correct_dict[method_name1][1]+1]
        correct_dict[method_name2] = [correct_dict[method_name2][0]  , correct_dict[method_name2][1]+1]
        method1_fail_counter += 1
        method1_fail_dict[input_name] = vote_dict[input_human_]
        # save_image = True
    elif max_key == '2':
        # correct_dict[method_name1] = [correct_dict[method_name1][0]+1, correct_dict[method_name1][1]+1]
        # correct_dict[method_name2] = [correct_dict[method_name2][0]  , correct_dict[method_name2][1]+1]
        correct_dict[method_name1] = [correct_dict[method_name1][0]  , correct_dict[method_name1][1]+1]
        correct_dict[method_name2] = [correct_dict[method_name2][0]+1, correct_dict[method_name2][1]+1]
        method2_fail_counter += 1
        method2_fail_dict[input_name] = vote_dict[input_human_]
        # save_image = True
    elif max_key == '3':
        correct_dict[method_name1] = [correct_dict[method_name1][0]+1, correct_dict[method_name1][1]+1]
        correct_dict[method_name2] = [correct_dict[method_name2][0]+1, correct_dict[method_name2][1]+1]
        both_succeed_counter += 1
        both_succeed_dict[input_name] = vote_dict[input_human_]

    
    if save_image:
        huamn_path = input_human_.replace(url_root, data_root)
        concat_paths = [huamn_path]+[item.replace(url_root, data_root) for item in vote_dict[input_human_]['options']]
        concat_titles = ['human']+[method_name1, method_name2]
        im_concat = data_utils.concat_list_image(concat_paths,concat_titles)
        save_dir = label_csv.replace('.csv', '')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, input_name+'.png')
        cv2.imwrite(save_path, im_concat)

# sort correct_dict by key
# correct_dict = dict(sorted(correct_dict.items(), key=lambda item: item[0]))

for key in correct_dict:
    print(key, round(100*correct_dict[key][0]/correct_dict[key][1],2), 'countes: ', correct_dict[key][0], correct_dict[key][1])

# debug_list.sort()
# print(debug_list)

print('Tag: method1_better_counter: ', method1_fail_counter)
print('Dir: method2_better_counter: ', method2_fail_counter)
print('both_failed_counter: ', both_failed_counter)
print('both_succeed_counter: ', both_succeed_counter)
a=1