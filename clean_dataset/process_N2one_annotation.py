import csv
import json
import data_utils
from PIL import Image
import cv2
import numpy as np
import os

root = '/Users/bytedance/Desktop/data/annotations/811_fairV3_430_nto1/'
csv_path = root+'annotation_translate.csv'


asset_dir = '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/bitmoji asset_version3'
task_img_dir = root + '811_image'
option_asset_dir = root+'811_asset_json'
option_asset_paths = data_utils.make_json_set(option_asset_dir)

bitmoji_auto_dir = '/Users/bytedance/Desktop/data/annotations/benchmark_v3/matched_results/3_bitmoji_auto'
bitmoji_auto_paths = data_utils.make_im_set(bitmoji_auto_dir)
bitmoji_manual_dir = '/Users/bytedance/Desktop/data/annotations/benchmark_v3/matched_results/4_bitmoji_manual'
bitmoji_manual_paths = data_utils.make_im_set(bitmoji_manual_dir)
invalid_path = '/Users/bytedance/Desktop/data/annotations/811_fairV3_430_nto1/invalid.png'

save_dir = root+'mapped'

annotation_round = 3
option_count = 5


basic_info = ['\ufeffTask ID','Object ID']
round_info = ['Handling Time','Record']
record_info = ['score','length','curly','braid','bang']


def load_record(info_string):
    info_list = json.loads(row[round_name+key])['mainForm']
    out_dict = {}
    for info in info_list:
        if info['name'] =='dummy':continue
        option = 'option'+info['name'][-1]
        if option not in out_dict : out_dict[option] = {} 
        var_name = info['name'][:-1]
        
        out_dict[option][var_name] = int(info['value']) if len(info['value'])>0 else 0
        
    return out_dict

def concat_list_image(matched_asset_paths,matched_titles=None):
    assert len(matched_asset_paths) > 0, "No matched asset found"
    #use cv2
    imgs = [cv2.imread(str(asset_path)) for idx, asset_path in enumerate(matched_asset_paths)]
    
    try:
        height_list = [img.shape[0] for img in imgs]
    except:
        a=1
    max_height = max(height_list)
    width_list = [img.shape[1] for img in imgs]
    max_width = max(width_list)
    resized_imgs = []
    
    for idx, img in enumerate(imgs):
        ratio = max_height/img.shape[0]
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        img = cv2.putText(img, matched_titles[idx], (20,img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 8)
        if idx == 0:
            im_concat = img
        else:
            im_concat = np.concatenate((im_concat, img), axis=1)
    return im_concat

# load option assets 
option_asset_dict = {}
for option_asset_path in option_asset_paths:
    option_asset_dict[option_asset_path.split('/')[-1].split('.')[0]] = json.load(open(option_asset_path))

data_dict = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_dict = {}

        for key in basic_info:
            try:
                row_dict[key.replace('\ufeff', '')] = row[key].replace('\t', '')
            except:
                row_dict[key.replace('\ufeff', '')] = row[key.replace('\ufeff', '')].replace('\t', '')

        
        row_dict['task_name']    = row['Object ID'].split('/')[-1].split('.')[0]
        row_dict['task_options'] = option_asset_dict[row_dict['task_name']]
        row_dict['task_image_name'] = row_dict['task_name'].replace('811_nto1_','')

        for i in range(annotation_round):
            round_name = str(i+1)+"Round"
            for key in round_info:
                if 'Record' in key:
                    row_dict[round_name+key] = load_record(row[round_name+key])#json.loads(row[round_name+key])
                else:
                    row_dict[round_name+key] = row[round_name+key]
        data_dict[row_dict['Task ID']] = row_dict
a=1

save_dict = {}
for task in data_dict:
    a=1
    save_dict[task] = {}
    task_name = data_dict[task]['task_name']
    image_name = data_dict[task]['task_image_name']


    for i in range(annotation_round):
        round_asset = {}
        round_name = str(i+1)+"Round"
        save_dict[task][round_name] = {}

        recod = data_dict[task][round_name+'Record']
        scores = [ recod[item]['score'] for item in recod]
        scores_idx = range(len(scores))
        # scores_sorted, scores_idx_sorted = zip(*sorted(zip(scores, scores_idx), reverse=True))
        
        for score, idx in zip(scores,scores_idx):
            if score == 0: continue
            round_asset[data_dict[task]['task_options'][idx]] = scores[idx]
            # round_asset.append(data_dict[task]['task_options'][idx])
        round_asset = dict(sorted(round_asset.items(), key=lambda item: item[1],reverse=True))
        save_dict[task][round_name] = round_asset
        a=1

    matched_asset_paths = [task_img_dir+'/'+task_name+'.jpg']
    matched_titles = ['input']
    aggre_votes = {}
    for i in range(annotation_round):
        round_name = str(i+1)+"Round"
        for matches in save_dict[task][round_name]:
            score = save_dict[task][round_name][matches]
            aggre_votes[matches] = score if matches not in aggre_votes else aggre_votes[matches] + score
            image_path = asset_dir+'/'+matches
            matched_asset_paths.append(image_path)
            break
            # asset_im = Image.open(str(image_path))
        matched_titles.append("Round"+str(i+1))

    unique = set(matched_asset_paths)
    options = {}
    for i in range(annotation_round):
        options[str(i+1)] = matched_asset_paths[i+1]
    disagree = 1 if len(unique)>2 else 0

    # Aggreated votes: Include the max score asset
    max_key = asset_dir+'/'+max(aggre_votes, key=aggre_votes.get)
    matched_asset_paths.insert(1,max_key)
    matched_titles.insert(1,"Aggre")

    if bitmoji_auto_dir+'/'+image_name+'.jpg' in bitmoji_auto_paths:
        cur_bitmoji_auto_path = bitmoji_auto_dir+'/'+image_name+'.jpg'
        cur_bimojit_manual_path = bitmoji_manual_dir+'/'+image_name.split('_')[0]+'.jpg'
    else:
        cur_bitmoji_auto_path = invalid_path
        cur_bimojit_manual_path = invalid_path
    matched_asset_paths.append(cur_bitmoji_auto_path)
    matched_titles.append("Auto")
    matched_asset_paths.append(cur_bimojit_manual_path)
    matched_titles.append("Manual")

    im_concat = concat_list_image(matched_asset_paths,matched_titles)

    cont_save_dir = str(save_dir)+'_concatenate'
    os.makedirs(cont_save_dir, exist_ok=True)
    cv2.imwrite(str(cont_save_dir+'/'+task_name+'.png'), im_concat)
    
    # if disagree:
    disagree_asset_paths = [task_img_dir+'/'+task_name+'.jpg']
    disagree_titles = ['input']
    for i in range(option_count):
        cur_asset = asset_dir+'/'+data_dict[task]['task_options'][i]
        cur_title = ''
        for option in options:
            if cur_asset in options[option] : 
                cur_title += ', '+option if  cur_title != '' else 'R :'+option

        disagree_asset_paths += [cur_asset]
        disagree_titles      += [cur_title]

    disagree_asset_paths.append(cur_bitmoji_auto_path)
    disagree_titles.append("Auto")
    disagree_asset_paths.append(cur_bimojit_manual_path)
    disagree_titles.append("Manual")

    im_disagree_concat = concat_list_image(disagree_asset_paths,disagree_titles)
    cont_disagree_save_dir = str(save_dir)+'_disagree'
    os.makedirs(cont_disagree_save_dir, exist_ok=True)
    cv2.imwrite(str(cont_disagree_save_dir+'/'+task_name+'.png'), im_disagree_concat)

            

    
a=1
