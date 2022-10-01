


prior_labels = ['01_clean','02_blur','03_hat','04_multiple','05_no_person']
target_label = None #'01_clean'

# Define label attributes
base_row_titles =   ['\ufeff任务ID','对象ID'     ,'进审时间'    ,'结束时间']
base_dict_titles =  ['id'         ,'image_url'  ,'task_start','task_end']

round_base_row_titles  = ['审核人','分配时间','解决时间','审核时长' ,'审核记录']          # Round Base info
round_base_dict_titles = ['id'   ,'start'  ,'end'   ,'duration','record']          # Round Base info
round_quality_titles = ['blur','mutiperson','head_occlusion']                      # Round quality
round_attri_one_titles = ['shorthair','curly','braid']                             # Round one
round_attri_two_titles = ['bang_length','hairstyle_length','hairstyle_curly']      # Round two
round_row_titles  =  round_base_row_titles  + round_quality_titles+round_attri_one_titles+round_attri_two_titles
round_dict_titles =  round_base_dict_titles + round_quality_titles+round_attri_one_titles+round_attri_two_titles

# Chinese to int value mapper
terms_val_mapper = {
    'blur':{'模糊，影响':0,'稍微模糊（不影响）':1,'不模糊':2},
    'mutiperson':{'多人，主角不明':0,'多人(主角明确)':1,'无多人':2},
    'head_occlusion':{'是，影响':0,'有帽子（不影响）':1,'无遮挡':2},
    'shorthair': { "是": 0, "否": 1 },
    'curly': { "是": 0, "否": 1 },
    'braid': { "是": 0, "否": 1 },
    'bang_length': { "无刘海": 0, "一点点刘海": 1, "不全遮挡刘海": 2, "遮全额头刘海": 3 },
    'hairstyle_length': { "光头": 0, "板寸": 1, "短发": 2, '中等长度': 3, '中长发': 4, '披肩': 5, '长发': 6 },
    'hairstyle_curly': { "无纹理": 0, "有纹理": 1, "直发": 2, "微卷": 3, "大卷": 4, "脏辫卷": 5 },
}


string2int = ['duration']


# Convert csv file to json file (Used on GUI))
# chinese to english value mapper
terms_val_mapper_GUI = {
    'shorthair': { "是": "0-Short", "否": "1-Not Short" },
    'curly': { "是": "0-Curl", "否": "1-Not Curl" },
    'braid': { "是": "0-Braid", "否": "1-Not Braid" },
    'bang_length': { "无刘海": "0-none", "一点点刘海": "1-little", "不全遮挡刘海": ["1-little","2-parting"], "遮全额头刘海": "3-full" },
    'hairstyle_length': { "光头": "0-bald", "板寸": "1-crew_cut", "短发": "2-short", '中等长度': "3-medium", '中长发': "4-medium_plus", '披肩': "5-shoulder", '长发': '6-long','暂缺':"7-braids" },
    'hairstyle_curly': { "无纹理": "1-misc", "有纹理": "2-directional", "直发": "3-straight", "微卷": "4-curly", "大卷": "5-curly_plus", "脏辫卷": "6-coiled" },
}

round_quality_titles_GUI = ['blur','mutiperson','head_occlusion']                      # Round quality
round_attri_one_titles_GUI = ['If short','If curl','If braid']                             # Round one
round_attri_two_titles_GUI = ['bang','length','texture']      # Round two
round_dict_titles_GUI =  round_base_dict_titles + round_quality_titles_GUI + round_attri_one_titles_GUI + round_attri_two_titles_GUI