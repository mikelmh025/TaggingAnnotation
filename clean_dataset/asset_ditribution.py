from cmath import pi
import json
import data_utils
import process_utils
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle


root = '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/'
json_dir = root+'bitmoji asset_version3_label'

json_dir2 = '/Users/bytedance/Desktop/data/annotations/aggregated3'

json_paths = data_utils.make_json_set(json_dir)
json_paths2 = data_utils.make_json_set(json_dir2)

terms_val_mapper_GUI = process_utils.terms_val_mapper_GUI
round_attri_two_titles = ['braid']+process_utils.round_attri_two_titles
round_attri_two_titles_GUI = ['If braid']+process_utils.round_attri_two_titles_GUI


attribute_dict = {}
for idx,key in enumerate(round_attri_two_titles):
    vals = terms_val_mapper_GUI[key]
    val_list = [-1]
    for val_key in vals:
        if isinstance(vals[val_key],list):        
            val_list.append(int(vals[val_key][-1].split("-")[0]))
        else:
            val_list.append(int(vals[val_key].split("-")[0]))
        

    attribute_dict[round_attri_two_titles_GUI[idx]] = val_list

attri_names = [key for key in attribute_dict]

attribute_dict_ = [attribute_dict[key] for key in attribute_dict]
attribute_permute = list(itertools.product(attribute_dict_[0], attribute_dict_[1], attribute_dict_[2],attribute_dict_[3]))

over_all_dict = {}
over_all_dict2 = {}
for item in attribute_permute:
    over_all_dict[item] = 0
    over_all_dict2[item] = 0

# Get asset distribution
for json_path in json_paths:
    json_dict = json.load(open(json_path))
    encode_val,val_name = [],[]
    for idx,_ in enumerate(round_attri_two_titles_GUI):
        title = round_attri_two_titles_GUI[idx]
        assert title in json_dict
        if len(json_dict[title]) != 1:
            a=1

        cur_encode_val =  [int(item.split("-")[0]) for item in json_dict[title]] + [-1]
        encode_val.append(cur_encode_val)
        val_name.append(title)
    # encode_val = tuple(encode_val)
    cur_permutes = list(itertools.product(encode_val[0], encode_val[1], encode_val[2],encode_val[3]))

    for permute in cur_permutes:
        over_all_dict[permute] += 1
    # if encode_val in attribute_permute:

    a=1

# Get human distribution
for json_path in json_paths2:
    json_dict = json.load(open(json_path))
    encode_val,val_name = [],[]
    for idx,_ in enumerate(round_attri_two_titles_GUI):
        title = round_attri_two_titles_GUI[idx]
        assert title in json_dict
        if len(json_dict[title]) != 1:
            a=1

        cur_encode_val =  [int(item.split("-")[0]) for item in json_dict[title]] + [-1]
        encode_val.append(cur_encode_val)
        val_name.append(title)
    # encode_val = tuple(encode_val)
    cur_permutes = list(itertools.product(encode_val[0], encode_val[1], encode_val[2],encode_val[3]))

    for permute in cur_permutes:
        over_all_dict2[permute] += 1
    # if encode_val in attribute_permute:

    a=1


def encode_dict_plot(encode_dict,braid_target=0,title=''):
    xdata,ydata,zdata,count = [],[],[],[]

    for key,val in encode_dict.items():
        if val == 0: continue
        braid,x,y,z = key
        if braid ==-1 or x == -1 or y == -1 or z == -1: continue
        if braid != braid_target: continue

        xdata.append(x)
        ydata.append(y)
        zdata.append(z)
        count.append(val)

        # print(key,val)
        # print(key,encode_dict[key])
    xdata,ydata,zdata,count = np.array(xdata),np.array(ydata),np.array(zdata),np.array(count)
    
    # Softmax count
    count = 1/(1 + np.exp(-count))*10

    ax = plt.axes(projection='3d')
    # ax.scatter3D(xdata, ydata, zdata, c=count, cmap='Greens')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    # ax.scatter3D(xdata, ydata, zdata)
    # plt.scatter3D(xdata,ydata, c=count, alpha=0.5)
    # plt.show()
    ax.set_xlabel('bang')
    ax.set_ylabel('length')
    ax.set_zlabel('texture')

    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()
    # a=1
    
        
encode_dict_plot(over_all_dict,braid_target=1,title='not_briad_asset_distribution')
encode_dict_plot(over_all_dict,braid_target=0,title='briad_asset_distribution')

# Save distribution dict to json
json_path = 'asset_distribution.pkl'
out_dict = {
    'asset_data':over_all_dict,
    'human_data':over_all_dict2,
    'attri_names':attri_names,
    'attribute_dict':attribute_dict,
}
pickle.dump(out_dict,open(json_path,'wb'))
# with open(json_path,'w') as f:
    # json.dump(out_dict,f)

for key in over_all_dict:
    vals = list(key)
    out_encode_name = ""
    for name in attri_names:
        out_encode_name += name+":"+str(vals[attri_names.index(name)])+" "

    asset_count, human_count = over_all_dict[key], over_all_dict2[key]
    if human_count > 4 and asset_count == 0:
        print(out_encode_name+"\t Asset count:"+str(over_all_dict[key]),"\t Human Count:"+str(over_all_dict2[key]))
a=1