import data_utils
import os
import csv
import random

root = '/Users/minghaoliu/Desktop/Data_HITL_navi/'

url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/Data_HITL_navi/'

human_dir = 'test'
# mode = 'matching'
# mode = 'subjective'
mode = ['matching','subjective']

match_options = 4

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
    'test_direct_bd':['test_mapped2'],
    'test_direct_pred':['pred'],
    'test_direct_turk':['Turk 3'],
    'test_other1_pred':[],
    'test_other2_pred':[],
    'test_other3_pred':[],
    'test_tag_bd':['1','aggre'],
    'test_tag_pred':['top1','top2','top3'],
    'test_tag_turk':['_match3','_match_aggre']
}



human_image_paths = data_utils.make_im_set(os.path.join(root, human_dir))
# for human_image_path in human_image_paths:
#     human_image_name = os.path.basename(human_image_path)

if 'subjective' in mode:
    combined_csv_path = os.path.join(root, csv_save_dir, 'combined_subjective.csv')
    with open(combined_csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_person','image_cartoon'])

    for target_dir in target_dir_dict:  # IE. test_direct_bd
        for target_subdir in target_dir_dict[target_dir]:  # IE. test_mapped1
            target_image_paths = data_utils.make_im_set(os.path.join(root, target_dir, target_subdir))

            # Create csv file
            csv_path = os.path.join(root, csv_save_dir, target_dir+ '_' + target_subdir + '_subjective.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['image_person','image_cartoon'])

            for target_image_path in target_image_paths:
                human_name = target_image_path.split('/')[-1].split('_')[0]
                target_name = target_image_path.split('/')[-1].split('_')[1].replace('.jpg','')

                human_path_ = os.path.join(root, human_dir, human_name+'.jpg')
                assert os.path.exists(human_path_), human_path_
                assert os.path.exists(target_image_path), target_image_path
                human_url_ = human_path_.replace(root, url_root)
                target_url_ = target_image_path.replace(root, url_root)

                # Write to individual CSV
                with open(csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([human_url_, target_url_])
                # Write to combined CSV
                with open(combined_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([human_url_, target_url_])
            # break

if 'matching' in mode:
    template_dict = {}
    combined_csv_path = os.path.join(root, csv_save_dir, 'combined_matching.csv')
    with open(combined_csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_person','image_cartoon'])

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
                writer.writerow(['image_person','image_url1','image_url2','image_url3','image_url4','image_url5','image_url6','image_url7','image_url8','image_url9','image_url10'])
                
            # Process each human image / Case
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
                    random.shuffle(other_image_paths)

                    other_image_paths = other_image_paths[:match_options-1]

                    all_options = [target_url_] + [other_image_path.replace(root, url_root) for other_image_path in other_image_paths]
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

                # Write to individual CSV
                with open(csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(all_options)
                # Write to combined CSV
                with open(combined_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(all_options)

        # break
