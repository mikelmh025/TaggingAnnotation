from logging import root
import pickle
import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import math

# Load pickle
root = '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/'
pickle_path = root+'bitmoji asset_version3_label_distribution.pkl'
with open(pickle_path,'rb') as f:
    data_dict = pickle.load(f)

assset_distribution = data_dict['asset_data']
human_distribution = data_dict['human_data']
attri_names = data_dict['attri_names']
attribute_dict = data_dict['attribute_dict']

def random_generate_encode(attribute_dict):
    out = []
    for attri_name in attribute_dict:
        out += np.random.choice(attribute_dict[attri_name],size=1).tolist()
    return out

def eval_distance(case1, case2):
    score_step= [10,1,2,2] # This is manully set by Minghao, need to tune this value by human annotaiton 
    score_level = [2,4,7,6] 
    score = []
    assert len(case1) == len(case2), "case1 and case2 must have the same length"
    for idx,item in enumerate(case1):
        cur_1, cur_2 = case1[idx], case2[idx]
        max_score = score_step[idx]*(score_level[idx]-1)

        if cur_1 == -1 : # original case with no constraint
            cur_step_diff = 1
        elif cur_2 == -1: # Target case with no constraint
            cur_step_diff = score_level[idx]-1
        else:
            cur_step_diff = abs(cur_1-cur_2)

        cur_score = max_score - cur_step_diff*score_step[idx]
        score.append(cur_score)
    return score
        

for i in range(10):
    sample = random_generate_encode(attribute_dict)
    if assset_distribution[tuple(sample)] > 0: continue
    target_encode_list,score_list = [],[]
    for key, val in assset_distribution.items():
        if val == 0:continue
        scores = eval_distance(sample,list(key))
        # print("case1:",sample,"case2:",list(key),"score:",scores)
        score_sum = sum(scores)
        target_encode_list.append(list(key))
        score_list.append(score_sum)
        # braid,x,y,z = key
    # Sort by score
    target_encode_list,score_list = zip(*sorted(zip(target_encode_list,score_list),key=lambda x:x[1],reverse=True))
    
    print("sample",sample,"Sample matched asset count: ",assset_distribution[tuple(sample)])
    for idx,item in enumerate(target_encode_list):
        print("target_encode_list:",target_encode_list[idx],"score_list:",score_list[idx],"asset_count:",assset_distribution[tuple(target_encode_list[idx])])
        if idx >= 10:break


    print(sample)


a=1
