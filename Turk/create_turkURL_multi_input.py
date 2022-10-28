import data_utils
import csv
import cv2
import os
import numpy as np
from typing import Tuple


save_img_root = '/Users/minghaoliu/Desktop/Data_HITL_navi/other_system/'
save_img_dir = 'google_cartoon/'
# save_img_dir = 'metahuman/'

url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/Data_HITL_navi/other_system/'
# url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/test/'

img_dir = '/Users/minghaoliu/Desktop/Data_HITL_navi/other_system/cartoonset100k/save_dir/0110'
# img_dir = '/Users/minghaoliu/Desktop/Data_HITL_navi/other_system/metahuman/metahuman'

save_root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/'
csv_path = save_img_dir.replace('/','') + '.csv'
img_paths = data_utils.make_im_set(img_dir)
img_paths.sort()



def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

# save to csv
with open(save_root+csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_url1'])

    buffer, buff_id = [], ''
    for img_path in img_paths:
        id = img_path.split('/')[-1].split('_')[0]
        if id != buff_id and len(buffer)>0:
            # print("clear buffer and save results")
            matched_titles = ['']*len(buffer)
            im_concat = data_utils.concat_list_image(buffer,matched_titles)
            
            save_path = os.path.join(save_img_root+save_img_dir, buff_id+'.jpg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            im_concat = resize_with_pad(im_concat, (1000, 1000))

            cv2.imwrite(save_path, im_concat)
            buffer, buff_id = [img_path], id
            url = save_path.replace(save_img_root, url_root)
            writer.writerow([url])
        else:
            buffer.append(img_path)
            buff_id = id

    a=1
        # name = img_path.split('/')[-1]
        # url = url_root + name
        # writer.writerow([url])