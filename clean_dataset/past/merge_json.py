import json
import data_utils

root = '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/'
json_dir1 = root + 'bitmoji asset_version2_label3'
json_dir2 = root + 'bitmoji asset_version2_label4'

assert json_dir1 == '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/bitmoji asset_version2_label3'
save_dir = root + 'bitmoji asset_version2_label5'


json_paths1 = data_utils.make_json_set(json_dir1)
json_paths2 = data_utils.make_json_set(json_dir2)

target_dir1 = ['If curl', 'If braid', 'texture', 'bang']
target_dir2 = ['If short', 'length']

for json_path1, json_path2 in zip (json_paths1,json_paths2):
    out_dict = {}
    # Load json
    json_dict1 = json.load(open(json_path1))
    json_dict2 = json.load(open(json_path2))
    print(json_dict1.keys())

    for key in json_dict1:
        if key in target_dir1 : out_dict[key] = json_dict1[key]
        if key in target_dir2 : out_dict[key] = json_dict2[key]
    # Merge json

    # Save json
    with open(save_dir+'/'+json_path1.split('/')[-1], 'w') as f:
        json.dump(out_dict, f)
    print(json_path1.split('/')[-1])
    a=1