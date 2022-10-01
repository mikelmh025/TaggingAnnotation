import data_utils
import cv2
import numpy as np


input_img_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/images'
ass_img_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/images'

input_img_paths = data_utils.make_im_set(input_img_dir)[0:1]
ass_img_paths = data_utils.make_im_set(ass_img_dir)

# print(ass_img_paths)
id_list = []
for ass_img_path in ass_img_paths:
    # print(ass_img_path)
    id = str(ass_img_path.split('/')[-1].split('.')[0])
    id_list.append(id)
    # print(id)
    # print('<option value="%s">%s</option>'%(id,id))

print(id_list)



def concat_asset(input_img_paths, ass_img_paths,column=5):

    # Read images
    input_imgs = [data_utils.read_img(path) for path in input_img_paths]
    asset_imgs = []
    for asset_path in ass_img_paths:
        asset_imgs.append(data_utils.read_img(asset_path,height=512))

    # Create rows
    image_rows = []
    for i in range(len(asset_imgs)//column+1):
        image_rows.append(data_utils.horizontal_cat(asset_imgs[i*column:(i+1)*column],column))

    asset_concat = data_utils.vertical_cat(image_rows)

    # for input_img in input_imgs:
    #     concat_img = vertical_cat([input_img, asset_concat])
    #     cv2.imwrite('concat_img.jpg', concat_img)

    # save concat_img
    cv2.imwrite('concat_img.jpg', asset_concat)
    # return img


concat_asset(input_img_paths, ass_img_paths)