import requests
import logging
import uuid
import hashlib
import json
from time import time
import urllib3
from datetime import datetime, timedelta

import math
import csv
urllib3.disable_warnings()


class TcsClient:
    def __init__(self):
        self.access_key = "SRP2R6P1HV"
        self.access_secret = "RLRU8SHW1NIPDD8HDV1GRNJR4FT6SUTVQTA9TTZ30RVDHYMNW2"

    def get_headers(self):
        timestamp = str(int(time()))
        #nonce = uuid.uuid1().get_hex()
        nonce = uuid.uuid4().hex
        _list = [self.access_secret, timestamp, nonce]
        _list.sort()

        signature = hashlib.sha1(''.join(_list).encode('utf8')).hexdigest()

        headers = {
            'X-AccessKey': self.access_key,
            'X-Signature': signature,
            'X-Timestamp': timestamp,
            'X-Nonce': nonce
        }
        return headers

    def create_task(self, project_id,data_dict):

        url = data_dict['url']
        rsp = requests.post("https://tcs.bytedance.net/api/v2/create_task/", data=dict(
            project_id=str(project_id),
            object_id=url,
            object_data=json.dumps(data_dict

            )),
            verify=False, headers=self.get_headers()
        )

        print(rsp.content)
        body = json.loads(rsp.content)
        print(body["code"])
        print(body["message"])
        print(rsp.headers)


# csv_path ='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/round2/1_url.csv'
csv_path ='/Users/bytedance/Desktop/data/annotations/0_annotation/906_faceattribute_10k/906_faceattribute_10k_result3_quality_check_10.csv'


# project_id ="7140323100276671013" #第二批正式
project_id ="7141006776572314148"   #第二批质检


url_list = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cur_dict = {}
        for key in reader.fieldnames:
            cur_dict[key] = row[key]
        url_list+=[cur_dict]
        # url_list+=[row['url']]
# start_flag = 0
# idx = 0
for data_dict in url_list:
    # idx += 1
    # if data_dict['url'] == 'http://sf3-ttcdn-tos.pstatp.com/obj/douyin-video-storage/0906_fair_face_clean_48205.jpg':
    #     start_flag = 1
    # if start_flag == 0: continue

    # assert 'http://sf3-ttcdn-tos.pstatp.com/obj/douyin-video-storage' in url,'Make sure using external link / correct prefix'
    tcs_client = TcsClient()
    tcs_client.create_task(project_id, data_dict)


# list_url = ["http://tosv.byted.org/obj/avatar-creation/Blank%20diagram.png","http://tosv.byted.org/obj/avatar-creation/Blank%20diagram.png"]

# project_id ="7109765202534580750" # TCS 三个点， Queue ID

# for url in list_url:
# # url = "https://tosv.byted.org/obj/ttfe/9cc1000f013c081b236b_1583915985601.jpeg"
#     tcs_client = TcsClient()
#     tcs_client.create_task(project_id, url)


# Jimu 做template
# 上传 图片到 TOS bucket ==》 获得 URL
# 用这个script 生成 TCS task， 记得调整object id
# TCS 用template 生成 task
