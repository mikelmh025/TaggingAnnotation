import json
import data_utils
import cv2
import numpy as np
import os

root = '/Users/bytedance/Desktop/artistic_avatars/data/bitmoji/'
image_dir = root + 'bitmoji asset_version2'

# mkdir save_dir
save_dir = root + 'bitmoji asset_version2_clean'
os.makedirs(save_dir, exist_ok=True)

image_paths = data_utils.make_im_set(image_dir)

redundant_list = []

for image_path1 in image_paths:
    if image_path1 in redundant_list:
        continue

    mse_list, similar_paths = [],[]
    # Read image to numpy array
    image1 = cv2.imread(image_path1)
    height1, width1, channel1 = image1.shape
    # resize image
    # image1 = cv2.resize(image1, (256, 256))
    image_name = image_path1.split('/')[-1].split('.')[0]

    for image_path2 in image_paths:
        if image_path1 == image_path2: continue

        # Read image to numpy array
        image2 = cv2.imread(image_path2)
        height2, width2, channel2 = image2.shape
        if height1 != height2 or width1 != width2 or channel1 != channel2:
            continue
        # image2 = cv2.resize(image2, (256, 256))
        

        # Compare image by using mse
        mse = np.mean((image1 - image2) ** 2)
        if mse < 3:
            a=1
            mse_list.append(mse)
            similar_paths.append(image_path2)

        a=1

    cv2.imwrite(save_dir+'/'+image_path1.split('/')[-1], image1)
    
    redundant_list += similar_paths
    
    # # Find the index of the minimum value in mse_list
    # if len(mse_list) == 0: continue
    # min_mse = min(mse_list)
    # min_index = mse_list.index(min(mse_list))
    # # Get the image path of the minimum value
    # min_image_path = similar_paths[min_index]
    # min_image_save = cv2.imread(min_image_path)
    # # Save image
    
    # cv2.imwrite(save_dir+'/'+image_name+"_"+str(min_mse)+"_"+min_image_path.split('/')[-1], min_image_save)

        
