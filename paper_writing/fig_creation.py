import data_utils
import cv2
import os
import numpy as np
import json
import random

def create_template(header_list):
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

def create_paper_fig(root, method_dict, case_id=None, sub_plot_size=None,max_row=20):
    print('start creating figure')

    # Get all the image paths
    fair_face_tags = None
    input_image_paths = None
    method_paths_dict = {}
    for row_key in method_dict.keys():
        method_name = method_dict[row_key][0]
        asset_dir_ = os.path.join(root, method_name, method_dict[row_key][1])
        image_paths_ = data_utils.make_im_set(asset_dir_)
        # image_paths.sort()
        method_paths_dict[row_key] = {item.split('/')[-1].split('.')[0].split('_')[0]:item for item in image_paths_}
        # method_paths_dict[method_name].sort()
        if input_image_paths is None: input_image_paths = image_paths_

    # Only keep the cases for presentation
    if case_id is not None: 
        input_image_paths = [item for item in input_image_paths if item.split('/')[-1].split('.')[0].split('_')[0] in case_id]
        input_image_ids = [item.split('/')[-1].split('.')[0].split('_')[0] for item in input_image_paths]
        for item in case_id: 
            if item not in input_image_ids:
                print('case id {} not found'.format(item))
                raise



    if len(method_dict['col1']) == 4: 
        with open('/Users/minghaoliu/Desktop/Data_HITL_navi/test.json', 'r') as f: fair_face_tags = json.load(f)

    image_rows = []
    # Create each row
    for idx, input_image_path in enumerate(input_image_paths):
        input_image_name = input_image_path.split('/')[-1].split('.')[0]

        cur_row_paths, cur_header = [], []
        for row_key in method_paths_dict.keys():
            if len(method_dict[row_key]) == 4 and method_dict[row_key][3] != None: # Need to deal with gender
                same_gender = fair_face_tags[input_image_name+'.jpg'][2] == method_dict[row_key][3]
                if not same_gender: continue

            cur_row_paths += [method_paths_dict[row_key][input_image_name]] if input_image_name in method_paths_dict[row_key].keys() else ['/Users/minghaoliu/Desktop/HITL_navi/data/invalid.jpeg']
            cur_header += [method_dict[row_key][2]]

        # cur_row_paths1 = [method_paths_dict[row_key][input_image_name] for row_key in method_paths_dict.keys()]
        
        titles = [input_image_name] + ['']*(len(cur_row_paths)-1)
        img_row = data_utils.concat_list_image(cur_row_paths,sub_plot_size=sub_plot_size,matched_titles=titles)
        image_rows.append(img_row)
        header_list =  cur_header
    
    # Create the header for the figure
    # header_list = [method_dict[method][2] for method in method_dict.keys()]
    header_image = create_template(header_list)
    
    # Vertical Concatenate all the rows, each fig has max row of 20
    buffer,out_images = [header_image], []
    for idx, rows in enumerate(image_rows):
        buffer.append(rows)
        if len(buffer) == max_row or idx == len(image_rows)-1:
            out_images.append(data_utils.vertical_cat(buffer))
            buffer = [header_image]


        
    # out_image = data_utils.vertical_cat(image_rows)

    
    

    return out_images


def config(fig_id):


    

    if fig_id=='1':
        case_id = [9096, 35957,1965,29172, 57282, 21658, 21869,51632,
        84379, 17712, 44945, 80522,57130, 23699,5085,40465,63055,42058,
        63323, 23920, 69801, 21847,72774,39365,34899,29374, 31512,50488,
        17954,39459,23617,68993,77864,8467,46891,34546,81289, 65898,80633,
        77126,64628,44249,52629, 675,71553,34190,35926,39106,84652,79126,
        39502,26378, 65444, 65929, 10439, 66791, 82359, 20061, 20698,
        85595, 32048, 30688, 74223, 36158, 31179, 25332, 85806,41675,32466,
        4097,77660, 47628, 50802, 26823, 52195, 69329, 74972, 81086, 8736,
        38171, 71559, 59389, 48508, 58684, 70336, 32688, 68293, 39221, 
        50122, 14984, 80129,27397,36729,77749,20765, 6385,32808, 49712, 
        48430, 86573, 67028, 72356, 21281, 20982, 44569, 85719, 24882, 37182, 
        44391, 70430, 71706, 15533,9313,26258,1940, 71665, 15334, 15687, 23453, 
        76308, 68687, 11344,30981,8346]
        
        case_id = [str(item) for item in case_id]

        method_dict = {
            'col1':['test','','Input human'],
            'col2':['test_direct_bd','test_mapped1','Direct label'],
            'col3':['test_tag_bd','aggre_top1', 'Tag label Top1'],
            'col4':['test_tag_bd','aggre_top2', 'Tag label Top2'],
            'col5':['test_tag_bd','aggre_top3', 'Tag label Top3'],
            'col6':['test_tag_bd','aggre_top4', 'Tag label Top4'],
            'col7':['test_tag_bd','aggre_top5', 'Tag label Top5'],
        }
        
    elif fig_id=='3':
        case_id = [1965, 21658, 21869,51632,84379, 17712, 80522,57130, 23699,5085,63055,42058,
        63323, 29374,50488,17954,39459,77864,8467,46891,34546,81289, 65898,64628,44249,52629,
        34190,84652,26378, 65444, 10439,20061,20698, 85595, 30688, 74223, 36158, 
        31179, 25332, 85806,32466, 4097,77660, 47628, 50802, 26823, 52195, 69329, 81086, 8736,
        71559, 59389, 48508, 58684, 70336, 32688, 68293, 39221, 50122, 14984, 80129,36729,
        77749,20765, 6385, 67028, 72356, 21281, 44569, 24882, 37182, 44391, 
        70430, 71706, 15533,9313, 71665, 15334, 15687, 23453, 76308, 68687, 11344,30981,8346]
        case_id = [str(item) for item in case_id]
        method_dict = {
            'col1':['test','','Input human'],
            'col2':['test_tag_pred','top1', 'Our: Tag label Top1'],
            'col3':['test_direct_pred','resnet50_img_img','Direct pred'],
            'col4':['test_other_AgileAvatar_pred','pred', 'AgileAvatar'],
        }

    if fig_id=='7':

        case_id= [9096, 35957, 1965, 29172, 57282, 51632, 84379, 44945, 5085, 63323, 23920,
        21847, 39365, 31512, 17954, 39459, 34546, 80633, 675, 34190, 35926, 84652,
        79126, 65444, 65929, 10439, 82359, 20698, 85595, 32048, 74223, 
        36158, 31179, 85806, 32466, 4097, 77660, 69329, 8736, 71559, 48508, 70336, 
        39221, 50122, 80129, 36729, 77749, 20765, 20982, 37182, 44391, 70430, 
        15533, 9313, 26258, 1940, 71665, 15334, 76308, 30981]
        

        # T2: [34899, 29374,23617,68993,77864,8467,46891,65898,26378,25332,41675,47628,74972,27397,6385,44569,24882,68687]


        
        case_id = [str(item) for item in case_id]

        method_dict = {
            'col1':['test'                                  ,''          , 'Input human'     , None],
            'col2':['test_tag_bd'                           ,'aggre_top1', 'Bitmoji'         , None],
            'col3':['other_system/google_cartoon_results'   ,'matched'   , 'Google cartoon'  , None],
            'col4':['other_system/metahuman_female_results' ,'matched'   , 'Metahuman'       , 'Female'],
            'col5':['other_system/metahuman_male_results'   ,'matched'   , 'Metahuman'       , 'Male'  ],
            'col6':['other_system/NovelAI_male_results'     ,'matched'   , 'NovelAI'         , 'Male'  ],
            'col7':['other_system/NovelAI_female_results'   ,'matched'   , 'NovelAI'         , 'Female'],
        }


    return method_dict, case_id


if __name__ == "__main__":
    # fig_ids  = ['1','3','7']
    fig_ids  = ['7']

    for fig_id in fig_ids:
        data_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/'
        save_dir = 'paper_writing/Fig'+fig_id
        fig_name = 'test'   # Not including the extension

        method_dict, case_id = config(fig_id)

        # Randomly select 10 cases
        # case_id = random.sample(case_id, 10)

        # Need to pick the case I want
        # case_id = ['675','15687','8346','32808','37280'] # Limited case for development
        # case_id = None




        figs = create_paper_fig(data_root, method_dict,sub_plot_size=(256,256),case_id=case_id)
        for idx, fig in enumerate(figs):
            os.makedirs(save_dir,exist_ok=True)
            save_path = os.path.join(save_dir, fig_name+'_'+str(idx)+'.png')
            cv2.imwrite(save_path, fig)