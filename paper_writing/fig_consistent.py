import data_utils
import cv2
import os
import numpy as np
import json
import random

def create_template(header_list,mode=1):
    if mode==1:
        header_template = np.ones((25, 128, 3), dtype=np.uint8)*255
        font_scale, font_thickness = 0.33, 1
    elif mode==2:
        header_template = np.ones((50, 128, 3), dtype=np.uint8)*255
        font_scale, font_thickness = 0.75, 2
    elif mode==3:
        header_template = np.ones((50, 256, 3), dtype=np.uint8)*255
        font_scale, font_thickness = 0.75, 2
    header_images = []
    for header in header_list:
        img_header_ = header_template.copy()
        # get size of text
        line_width, line_height = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        left_space = (img_header_.shape[1] - line_width) // 2
        top_space = (img_header_.shape[0] - line_height) // 2
        # add text centered on image
        img_header_ = data_utils.add_text_to_image(img_header_,header,font_scale=font_scale,font_thickness=font_thickness, top_left_xy=(left_space,top_space),font_color_rgb=(0,0,0))
        header_images.append(img_header_)
    header_image = cv2.hconcat(header_images) 
    return header_image

def create_paper_fig(root, method_dict, case_id=None, sub_plot_size=None,max_row=20,get_topk=5):
    print('start creating figure')

    method_path_dict, method2title_dict = {},{}
    header_image = None

    for idx, row_key in enumerate(method_dict.keys()):
        method_name = method_dict[row_key][0]
        
        if idx == 0:
            input_key = method_name
            input_dir   = os.path.join(root, method_name)
            input_image_paths = data_utils.make_im_set(input_dir)
            method_path_dict_ =  input_image_paths
        else:
            method_path_dict_ = {}
            input_root   = os.path.join(root, method_name)
            for i in range(get_topk):
                input_dir   = os.path.join(input_root, 'top'+str(i+1))
                image_paths_ = data_utils.make_im_set(input_dir)
                method_path_dict_[str(i+1)] = {item.split('/')[-1].split('.')[0].split('_')[0]:item for item in image_paths_}
        method_path_dict[method_name] = method_path_dict_
        method2title_dict[method_name] = method_dict[row_key][1]
    if case_id is not None: 
        input_image_paths = [item for item in input_image_paths if item.split('/')[-1].split('.')[0].split('_')[0] in case_id]

    image_rows = []
    # Create each row
    # Create images using cur_dict
    for idx, input_image_path in enumerate(input_image_paths):
        input_image_name = input_image_path.split('/')[-1].split('.')[0]
        cur_dict,img_dict_ = {},{}
        for idx, method_name in enumerate(method_path_dict.keys()):
            if method_name == input_key:
                cur_dict[method_name] = [input_image_path]
                titles = [input_image_name]
                img_dict_[method_name] = data_utils.concat_list_image([input_image_path],sub_plot_size=sub_plot_size,matched_titles=titles)
            else:
                top_list, top_titles = [], []
                for i in range(get_topk):
                    top_list.append(method_path_dict[method_name][str(i+1)][input_image_name])
                    top_titles.append('top'+str(i+1))
                cur_dict[method_name] = top_list
                titles = ['']*len(top_list)
                
                img_dict_[method_name] = data_utils.concat_list_image(top_list,sub_plot_size=tuple([int(0.5*z) for z in sub_plot_size]) ,matched_titles=titles)

        # Create row image
        cur_img, input_key = None, None
         
        for i, key in enumerate(img_dict_):
            if i == 0: 
                input_key = key
                continue
            cur_method_img = img_dict_[key]
            sub_title_img = create_template([method2title_dict[key]])
            sub_title_img = cv2.rotate(sub_title_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cur_method_img = cv2.hconcat([sub_title_img,cur_method_img])

            cur_img = cur_method_img if cur_img is None else data_utils.vertical_cat([cur_img,cur_method_img])
            

        cur_img = cv2.hconcat([img_dict_[input_key],cur_img])
        image_rows.append(cur_img)

        # Create Header
        if header_image is None:
            header_image = create_template(top_titles,mode=2)
            input_header = create_template(['input'],mode=3)
            gap_header = np.ones([50,cur_img.shape[1] - header_image.shape[1] - input_header.shape[1],3],dtype=np.uint8)*255
            header_image = cv2.hconcat([gap_header,header_image])
            header_image = cv2.hconcat([input_header,header_image])
            cur_img = data_utils.vertical_cat([header_image,cur_img])

        # cv2.imwrite('cur_img.png',cur_img)
        a=1
    # Vertical Concatenate all the rows, each fig has max row of 20
    buffer,out_images = [header_image], []
    for idx, rows in enumerate(image_rows):
        buffer.append(rows)
        if len(buffer) == max_row or idx == len(image_rows)-1:
            out_images.append(data_utils.vertical_cat(buffer))
            buffer = [header_image]
    return out_images



def config(fig_id):


    if fig_id=='5':
        case_id = [1965, 21658, 21869,51632,84379, 17712, 80522,57130, 23699,5085,63055,42058,
        63323, 29374,50488,17954,39459,77864,8467,46891,34546,81289, 65898,64628,44249,52629,
        34190,84652,26378, 65444, 10439,20061,20698, 85595, 30688, 74223, 36158, 
        31179, 25332, 85806,32466, 4097,77660, 47628, 50802, 26823, 52195, 69329, 81086, 8736,
        71559, 59389, 48508, 58684, 70336, 32688, 68293, 39221, 50122, 14984, 80129,36729,
        77749,20765, 6385, 67028, 72356, 21281, 44569, 24882, 37182, 44391, 
        70430, 71706, 15533,9313, 71665, 15334, 15687, 23453, 76308, 68687, 11344,30981,8346]
        case_id = [str(item) for item in case_id]
        method_dict = {
            'col1':['test'            , 'Input human'],
            'col2':['test_tag_pred'   ,'Tag'   ],
            'col3':['test_direct_pred','Direct'],
        }
 

    return method_dict, case_id


if __name__ == "__main__":
    # fig_ids  = ['1','3','7']
    fig_ids  = ['5']

    for fig_id in fig_ids:
        data_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/'
        save_dir = 'paper_writing/Fig'+fig_id
        fig_name = 'test'   # Not including the extension

        method_dict, case_id = config(fig_id)

        # Randomly select 10 cases
        case_id = random.sample(case_id, 10)

        # Need to pick the case I want
        # case_id = ['675','15687','8346','32808','37280'] # Limited case for development
        # case_id = None




        figs = create_paper_fig(data_root, method_dict,sub_plot_size=(256,256),case_id=case_id)
        for idx, fig in enumerate(figs):
            os.makedirs(save_dir,exist_ok=True)
            save_path = os.path.join(save_dir, fig_name+'_'+str(idx)+'.png')
            cv2.imwrite(save_path, fig)