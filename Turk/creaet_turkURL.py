import data_utils
import csv

url_root = 'https://minghaouserstudy.s3.amazonaws.com/HITL_navi/test/'

img_dir = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/test'
save_root = '/Users/minghaoliu/Desktop/HITL_navi/Turk/turk_exp/'
csv_path = 'init_direct.csv'
img_paths = data_utils.make_im_set(img_dir)

# save to csv
with open(save_root+csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_url1'])
    for img_path in img_paths:
        name = img_path.split('/')[-1]
        url = url_root + name
        writer.writerow([url])