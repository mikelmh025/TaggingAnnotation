import os
import torch

MODEL_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG','png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in MODEL_EXTENSIONS)

# Only one level of file
def make_image_set(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # remove hidden files
            if fname.startswith('.'):
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
        break
    return images

def get_mean_std_all(data_loader):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    for i, (img, label,idx) in enumerate(data_loader):
        psum    += img.sum(axis        = [0, 2, 3]) / (img.shape[2]*img.shape[3])
        psum_sq += (img ** 2).sum(axis = [0, 2, 3]) / (img.shape[2]*img.shape[3])
    count = len(data_loader) #* 256 * 256
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)
    return total_mean, total_std


def attr2int(attr):
    try:
        out = int(attr.split('-')[0])
    except:
        out = float(attr.split('-')[0])
    return out

def intersection_list(list1,list2):
    return list(set(list1) & set(list2))

def intersection_list_keep_size(list1,list2):
    return [value for value in list1 if value in list2]

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return [num]

def concat_list_image(matched_asset_paths,matched_titles=None):
    assert len(matched_asset_paths) > 0, "No matched asset found"
    #use cv2
    imgs = [cv2.imread(str(asset_path)) for idx, asset_path in enumerate(matched_asset_paths)]
    
    try:
        height_list = [img.shape[0] for img in imgs]
    except:
        # throw error if image is not read
        raise Exception("Image not read")
    max_height = max(height_list)
    width_list = [img.shape[1] for img in imgs]
    max_width = max(width_list)
    resized_imgs = []
    
    for idx, img in enumerate(imgs):
        ratio = max_height/img.shape[0]
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        img = add_text_to_image(img,matched_titles[idx],font_scale=1,font_thickness=2, top_left_xy=(20,img.shape[0]-500))
        # img = cv2.putText(img, matched_titles[idx], (20,img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        if idx == 0:
            im_concat = img
        else:
            im_concat = np.concatenate((im_concat, img), axis=1)
    return im_concat


from typing import Optional, Tuple

import cv2
import numpy as np


def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: Tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: Tuple = (0, 0, 255),
    bg_color_rgb: Optional[Tuple] = None,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb

def fix_image_subscript(img_path):
    sub_scripts = ['.jpg','.JPG','.png','.PNG','.jpeg','.JPEG']
    if exists(img_path):
        return img_path
    else:
        subscirpt = '.'+img_path.split('.')[-1]
        path_name = img_path.replace(subscirpt,'')

        for option in sub_scripts:
            if exists(path_name+option):
                return path_name+option

        raise Exception("Image not read")


def horizontal_cat(imgs,column):
    if len(imgs) <column:
        imgs += [np.zeros_like(imgs[0])]*(column-len(imgs))
    for i in range(len(imgs)):
        concat_img = cv2.hconcat([concat_img, imgs[i]]) if i != 0 else imgs[i]
    return concat_img

def vertical_cat(imgs):
    min_width = max([img.shape[1] for img in imgs])

    for i in range(len(imgs)):
        ratio = min_width/imgs[i].shape[1]
        img_ = cv2.resize(imgs[i], None, fx=ratio, fy=ratio)

        concat_img = cv2.vconcat([concat_img, img_]) if i != 0 else img_
    return concat_img

def read_img(path,height=512):
    img = cv2.imread(path)
    ratio = height/img.shape[0]
    img = cv2.resize(img, None, fx=ratio, fy=ratio)         # reshpae img to 512 height
    return img