from numpy import random
import data_utils
# import pandas as pd
import os
import csv
import random

root = '/Users/bytedance/Desktop/data/annotations/811_fairV3_430_nto1/'
dir = root+'mapped_disagreeV3'
out_file = root+'mapped_disagreeV3_url.csv'
image_paths = data_utils.make_im_set(dir)
image_paths.sort()
url_root = 'https://minghaouserstudy.s3.amazonaws.com/bitmoji+annotationV3/'

for image_path in image_paths:
    image_name = image_path.split('/')[-1]
    url = url_root+image_name
    with open(out_file,'a') as csvfile:
        csvwriter = csv.writer(csvfile) 
        out = "=IMAGE("+'\"'+url+"\")"
        csvwriter.writerow([out])


# root = 'user_study/matching/'
# benchmark = 'interal0425/'
# root +=benchmark

# csv_paths = sorted(data_utils.make_dataset_csv(root))
# num_options = 1
# merge = 10

# for csv_path in csv_paths:
#     if 'google' in csv_path: continue
#     df = pd.read_csv(csv_path)
#     out_file = csv_path.replace('.csv','_google.csv')
#     if merge >0 : out_file = root+'merge.csv'
#     merge_cur = 0
#     for index, row in df.iterrows():
#         if merge_cur>=merge: break
#         a = random.uniform(0, 1)
#         if a <= 0.5: continue
        
#         empty = ['']
#         for i in range(num_options-2):
#             empty+=['']
#         data_rows = [row['image_url1'] ]+ empty if num_options >= 2 else [row['image_url1'] ]

#         answer = row['image_url1'].split('/')[-1].split('.')[0].split('_')[-1][-1]
#         answer_list = []
#         for i in range(num_options):
#             if int(i) == int(answer):
#                 answer_list.append(1)
#             else:
#                 answer_list.append('')

#         merge_cur+=1

#         with open(out_file,'a') as csvfile:
#             csvwriter = csv.writer(csvfile) 
            
#             for idx in range(len(data_rows)):
#                 csvwriter.writerow([data_rows[idx],bool(answer_list[idx]),answer])
                
