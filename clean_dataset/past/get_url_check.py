import data_utils
import csv

# dataset_dir='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/annotation_degbug/hair_sub_attribute_0713_batch1'
dataset_dir='/Users/bytedance/Desktop/data/annotations/820_faceattribute_round2/human_face'
dataset_dir='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/round2/0825_fair_face_clean'

# url_root = 'http://tosv.byted.org/obj/avatar-creation/'
# 外网URL "http://sf3-ttcdn-tos.pstatp.com/obj/douyin-video-storage"
# internal: http://tosv.byted.org/obj/douyin-video-storage/00000000.mp4
# external: http://sf3-ttcdn-tos.pstatp.com/obj/douyin-video-storage/00000000.mp4
url_root = 'http://sf3-ttcdn-tos.pstatp.com/obj/douyin-video-storage/'

img_paths = data_utils.make_im_set(dataset_dir)



url_list = []
for img_path in img_paths:
    name = img_path.split('/')[-1]
    url = url_root+name
    url_list.append(url)
    # print(url)


with open(dataset_dir+'_url.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['url'])
    for url in url_list:
        writer.writerow([url])

# Read CSV file

csv_path = ''
url_list = []
with open(dataset_dir+'_url.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        url_list+=[row['url']]
    # data = list(reader['url'])

a=1