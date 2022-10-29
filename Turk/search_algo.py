import data_utils
import json
import collections
import math
import os

class search_algorithm ():
    def __init__(self):
        # print("Initalizing search algorithm. Verion 2.0: 10/25/2022")
        print("Initalizing search algorithm. Verion 3.3: 10/27/2022")
        self.distance_score_quality= {
            'blur': 0, 
            'head_occlusion': 0, 
            'mutiperson': 0
            }

        # Consider non-linear disntace for each level
        # Consider some random number subscript for each level
        # From version 2.0 and 3.0
        # self.distance_score_linear = {
        #     'top_curly': 1, 
        #     'top_length': 1.5, 
        #     'side_curly': 2, 
        #     'side_length': 2, 
        #     'braid_count': 8, 
        #     }

        # self.distance_score_type = {
        #     'top_direction': 0, 
        #     'braid_tf': 10, 
        #     'braid_type': 3, 
        #     'braid_position': 2}
        
        # From version 3.1
        self.distance_score_linear = {
            'top_length': 2.25,
            'side_length': 2.25,
            'top_curly': 1,  
            'side_curly': 1, 
            }

        self.distance_score_type = {
            'braid_count': 2, 
            'top_direction': 2, 
            'braid_tf': 5, 
            'braid_type': 1, 
            'braid_position': 1,
            }

        self.top_direction_group1     = ['0-向下','1-斜下','2-横向','3-斜上','6-向上（头发立起来）','7-向后梳头（长发向后梳，大背头，马尾）']
        self.top_direction_group2     = ['4-中分','5-37分']
        self.top_direction_group1_int = [ data_utils.attr2int(attr) for attr in self.top_direction_group1]
        self.top_direction_group2_int = [ data_utils.attr2int(attr) for attr in self.top_direction_group2]

    def attr_vote2_list(self, vote_dict):
        out_list = []
        for key in vote_dict:
            out_list += [data_utils.attr2int(key)]*vote_dict[key]
        return out_list

    def type_distance(self, human,asset):
        if len(human) == 0 and len(asset) == 0: return 0  # given both empty no distance    
        return int(len(data_utils.intersection_list(human,asset)) <1)

    def filter_votes(self, votes):
        if len(votes) == 0: return []
            
        counter = collections.Counter(votes)
        max_count = max(counter.values())
        counter = {key:val for key,val in counter.items()}
        # sort dict by value
        counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1],reverse=True)}
        resulting_votes = [key for key in counter if counter[key] == max_count]
        return resulting_votes

    def eval_distance_specific(self, human, asset, attr_care):
        # Note: only support side_curly and top_curly now
        distance_dict = {'total':0}
        for attr in human:
            if attr not in attr_care: continue
            
            distance_ = 0
            human_attris = self.attr_vote2_list(human[attr])
            asset_attris = self.attr_vote2_list(asset[attr])
            if attr in self.distance_score_linear:

                human_attris, asset_attris = data_utils.most_frequent(human_attris), data_utils.most_frequent(asset_attris)
                assert len(human_attris) == 1 and len(asset_attris) == 1
                unit_dis = math.ceil(abs(human_attris[0] - asset_attris[0])/2)
                distance_ = self.distance_score_linear[attr] * unit_dis # Linear distance: absolute difference
            elif attr in self.distance_score_type:

                # Top direction: split tagges into 2 groups,  
                if attr == 'top_direction':
                    assert False, "Not implemented yet"
                else:
                    # majority vote of asset_attris
                    human_attris, asset_attris = data_utils.most_frequent(human_attris), data_utils.most_frequent(asset_attris)
                    # human_attris, asset_attris = self.filter_votes(human_attris), self.filter_votes(asset_attris)
                    distance_ = self.distance_score_type[attr] * self.type_distance(human_attris,asset_attris) # Type distance: Type error when no intersection

            
            distance_dict[attr] = distance_
        for attr in distance_dict:
            distance_ = distance_dict[attr]
            distance_sum = distance_ if type(distance_) != list else sum(distance_)
            distance_dict['total'] += distance_sum
        # distance_dict['total'] = max(distance_dict[attr_care[0]],distance_dict[attr_care[1]])
        return distance_dict, distance_dict['total']

    def eval_distance(self,human,asset):
        # MANUAL fix: use top direction for long hair only 
        # flag_long_hair = True if data_utils.most_frequent(attr_vote2_list(human['side_length']))[0] >=3 else False

        distance_dict = {'total':0}
        for attr in human:
            if attr not in asset: continue
            
            distance_ = 0
            human_attris = self.attr_vote2_list(human[attr])
            asset_attris = self.attr_vote2_list(asset[attr])

            # From Version 3.0
            # if attr == 'top_direction':
            #     continue

            
            if attr in self.distance_score_quality:
                distance_ = self.distance_score_quality[attr] * self.type_distance(human_attris,asset_attris)# Type distance: Type error when no intersection


            # Linear distance: aggregate by average
            elif attr in self.distance_score_linear:
                # From version 2.0
                # mean_human, mean_asset = sum(human_attris)/len(human_attris), sum(asset_attris)/len(asset_attris)
                # distance_  = self.distance_score_linear[attr] * abs(mean_human - mean_asset) # Linear distance: absolute difference

                # From 3.1 When when aggregate doesn't work, conside all options are possible answers
                # human_attris, asset_attris = self.filter_votes(human_attris), self.filter_votes(asset_attris)
                # distance_ = self.distance_score_linear[attr] * self.type_distance(human_attris,asset_attris) 


                # From Version 3.0
                human_attris, asset_attris = data_utils.most_frequent(human_attris), data_utils.most_frequent(asset_attris)
                assert len(human_attris) == 1 and len(asset_attris) == 1
                # distance_ = self.distance_score_linear[attr] * self.type_distance(human_attris,asset_attris) # Type distance: Type error when no intersection
                
                # From Version 3.3: non linear distance with ceil
                # unit_dis = math.ceil(abs(human_attris[0] - asset_attris[0])/2)
                unit_dis = abs(human_attris[0] - asset_attris[0])#/2
                # From Version 3.4: allow one level of ability to be wrong
                # if abs(human_attris[0] - asset_attris[0])==1: 
                #     unit_dis = 1#0.7
                distance_ = self.distance_score_linear[attr] * unit_dis # Linear distance: absolute difference
            elif attr in self.distance_score_type:

                # Top direction: split tagges into 2 groups,  
                if attr == 'top_direction':
                    # MANUAL fix: use top direction for long hair only 
                    
                    # self.distance_score_type[attr] = 2#0.2 if not flag_long_hair else 3

                    # Version 3.2
                    human_attris_g1, human_attris_g2 = data_utils.intersection_list_keep_size(human_attris,self.top_direction_group1_int), data_utils.intersection_list_keep_size(human_attris,self.top_direction_group2_int)
                    asset_attris_g1, asset_attris_g2 = data_utils.intersection_list_keep_size(asset_attris,self.top_direction_group1_int), data_utils.intersection_list_keep_size(asset_attris,self.top_direction_group2_int)
                    human_attris_g1, human_attris_g2 = self.filter_votes(human_attris_g1), self.filter_votes(human_attris_g2)
                    asset_attris_g1, asset_attris_g2 = self.filter_votes(asset_attris_g1), self.filter_votes(asset_attris_g2)
                    td_error1 = self.distance_score_type[attr] * self.type_distance(human_attris_g1,asset_attris_g1)
                    td_error2 = self.distance_score_type[attr] * self.type_distance(human_attris_g2,asset_attris_g2)
                    distance_ = td_error1 + td_error2
                    # distance_ = min(td_error1 + td_error2, self.distance_score_type[attr]) # 3 is the max error for top direction

                    # # From version 2.0, 3.0 and 3.1
                    # # Split taggs into 2 groups
                    # human_attris_g1, human_attris_g2 = data_utils.intersection_list(human_attris,self.top_direction_group1_int), data_utils.intersection_list(human_attris,self.top_direction_group2_int)
                    # asset_attris_g1, asset_attris_g2 = data_utils.intersection_list_keep_size(asset_attris,self.top_direction_group1_int), data_utils.intersection_list_keep_size(asset_attris,self.top_direction_group2_int)
                    
                    # # only calculate the distance when human has tag in a group,
                    # # aggregate human and asset tags by majority vote
                    # # TODO: what if one side of the party doesn't have tag? idea 1: simply ignore this term in this case; idea 2: conside as a mistake tag
                    # if len(human_attris_g1) > 0 and len(asset_attris_g1) > 0:
                    #     human_attris_g1, asset_attris_g1 = data_utils.most_frequent(human_attris_g1), data_utils.most_frequent(asset_attris_g1)
                    #     distance_ += self.distance_score_type[attr] * self.type_distance(human_attris_g1,asset_attris_g1) # Type distance
                    # if len(human_attris_g2) > 0 and len(asset_attris_g2) > 0:
                    #     human_attris_g2, asset_attris_g2 = data_utils.most_frequent(human_attris_g2), data_utils.most_frequent(asset_attris_g2)
                    #     distance_ += self.distance_score_type[attr] * self.type_distance(human_attris_g2,asset_attris_g2) # Type distance

                    # # Manual fix : if not tag in group 1, then consider as a mistake tag
                    # if len(asset_attris_g1) == 0 and flag_long_hair:
                    #     distance_ += self.distance_score_type[attr] * 1
                    a=1
                else:
                    # From verion 2.0 and 3.0
                    # majority vote of asset_attris
                    human_attris, asset_attris = data_utils.most_frequent(human_attris), data_utils.most_frequent(asset_attris)

                    # From 3.1 When when aggregate doesn't work, conside all options are possible answers
                    # human_attris, asset_attris = self.filter_votes(human_attris), self.filter_votes(asset_attris)
                    distance_ = self.distance_score_type[attr] * self.type_distance(human_attris,asset_attris) # Type distance: Type error when no intersection

            
            distance_dict[attr] = distance_
            
        braid_error = 0
        length_error = 0
        curly_error = 0
        for attr in distance_dict:
            distance_ = distance_dict[attr]
            distance_sum = distance_ if type(distance_) != list else sum(distance_)
            if 'braid' in attr: 
                braid_error += distance_sum
            elif 'length' in attr:
                length_error += distance_sum
            elif 'curly' in attr:
                curly_error += distance_sum
            else:
                distance_dict['total'] += distance_sum
        
        # distance_dict['total'] += min(length_error,(self.distance_score_linear['top_length']+self.distance_score_linear['side_length'])*0.75)
        distance_dict['total'] += min(length_error,5)
        # distance_dict['total'] += min(curly_error,self.distance_score_type['side_curly']) 
        distance_dict['total'] += min(braid_error,self.distance_score_type['braid_tf']) # Threshold for braid error
        if 'ban' in asset: distance_dict['total'] += 1000

        return distance_dict, distance_dict['total']

    def search_all_assets(self, human_label, asset_data, attr_care=None):
        dis_scores = {}
        dis_reports = {}
        # Search all assets
        for asset_key in asset_data:
            if attr_care == None:
                dis_dict, dis_sum = self.eval_distance(human_label,asset_data[asset_key])
            else:
                dis_dict, dis_sum = self.eval_distance_specific(human_label,asset_data[asset_key],attr_care)
            dis_scores[asset_key] = dis_sum
            dis_reports[asset_key] = dis_dict
        dis_scores = dict(sorted(dis_scores.items(), key=lambda d:d[1]))
        return dis_scores, dis_reports

    # output out_dis_scores{asset_key: int dis_sum}, out_dis_reports{asset_key: dict dis_dict}
    def multi_round_search(self,human_label,asset_data,
                            attr_care = ['top_curly','side_curly'],search_top=15):
        # Inital round of search    
        dis_scores1, dis_reports1 = self.search_all_assets(human_label,asset_data)

        # Trim asset pool based on round one search
        round2_dict = {key:value for idx, (key, value) in enumerate(dis_scores1.items()) if idx < search_top}
        asset_data_trimed = {key:value for key, value in asset_data.items() if key in round2_dict.keys()}
            
        # Second round of search
        dis_scores2, dis_reports2 = self.search_all_assets(human_label,asset_data_trimed, attr_care=attr_care)

        min_score = min(dis_scores2.values())
        asset_data_trimed2 = {key:value for key, value in asset_data_trimed.items() if dis_scores2[key] == min_score}
        dis_scores3, dis_reports3 = self.search_all_assets(human_label,asset_data_trimed2)

        out_dis_scores, out_dis_reports = dis_scores3, dis_reports3

        return out_dis_scores, out_dis_reports

    def get_one_matched(self, human_label,asset_data,human_key=None,
                        human_image_dir=None,asset_dir=None, 
                        attr_care = ['top_curly','side_curly'],search_top=15, show_top=5):
        assert search_top>=show_top, 'search_top must be larger than show_top'

        image_name = human_key.split('.')[0]

        huamn_path = os.path.join(human_image_dir,human_key)
        if os.path.exists(huamn_path+'.png'):
            huamn_path = huamn_path+'.png'
        elif os.path.exists(huamn_path+'.jpg'):
            huamn_path = huamn_path+'.jpg'
        else:
            huamn_path = huamn_path

        # Inital round of search    
        dis_scores1, dis_reports1 = self.search_all_assets(human_label,asset_data)

        # Trim asset pool based on round one search
        round2_dict = {key:value for idx, (key, value) in enumerate(dis_scores1.items()) if idx < search_top}
        asset_data_trimed = {key:value for key, value in asset_data.items() if key in round2_dict.keys()}
            
        # Second round of search
        dis_scores2, dis_reports2 = self.search_all_assets(human_label,asset_data_trimed, attr_care=attr_care)

        min_score = min(dis_scores2.values())
        asset_data_trimed2 = {key:value for key, value in asset_data_trimed.items() if dis_scores2[key] == min_score}
        dis_scores3, dis_reports3 = self.search_all_assets(human_label,asset_data_trimed2)

        out_dis_scores, out_dis_reports = dis_scores3, dis_reports3

        # out_dis_scores, out_dis_reports = dis_scores2, dis_reports2

        matched_asset_paths = [os.path.join(asset_dir, key) for key in out_dis_scores.keys()]
        matched_asset_paths = matched_asset_paths[:show_top]


        out_paths, out_titles = [huamn_path], ['human']
        out_paths += matched_asset_paths
        out_titles += [str(round(value,2)) for value in out_dis_scores.values()]

        out_titles = ['']
        for case in out_dis_scores:
            
            r1_report = [attr + ' ' + str(round(dis_reports1[case][attr],2)) + ' \n' for attr in dis_reports1[case] if dis_reports1[case][attr] != 0 and attr not in attr_care] 
            r1_report = ''.join(r1_report)
            r2_report = [attr + ' ' + str(round(dis_reports2[case][attr],2)) + ' \n' for attr in dis_reports2[case] if dis_reports2[case][attr] != 0]
            r2_report = ''.join(r2_report)

            title_  = 'R1: \n' + r1_report + 'R2: \n' + r2_report
            out_titles.append(title_)


        return out_paths, out_titles, image_name, out_dis_reports

if __name__ == '__main__':
    print('search_algo.py')
    human_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/all_results_soft.json'
    human_sample_json = json.load(open(human_sample_json_path))

    asset_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/asset/820_faceattribute_round2_asset_translate_soft.json'
    asset_sample_json = json.load(open(asset_sample_json_path))
    human_limit = 50
    
    print("Error Outdated version")
    exit()
    algo = search_algorithm()
    matched_dict = {}
    for idx, key1 in enumerate(human_sample_json):
        if idx >= human_limit: break
            
        matched_dict[key1] = {}

        all_score_dict = {}
        all_distance_dict = {}
        for key2 in asset_sample_json:
            distance_dict, total = algo.eval_distance(human_sample_json[key1],asset_sample_json[key2])
            all_score_dict[key2] = [total,distance_dict]
            # all_distance_dict[key2] = distance_dict


            print(key1,key2,total)
            print(distance_dict)
            # break
        # break
        # get lowest 5 score from all_score_dict
        sorted_score_dict = sorted(all_score_dict.items(), key=lambda x: x[1][0])
        for match_idx, item in enumerate(sorted_score_dict[:5]):
            matched_dict[key1][match_idx] = {}
            matched_dict[key1][match_idx]['name'] = item[0]
            matched_dict[key1][match_idx]['score'] = item[1][0]
            matched_dict[key1][match_idx]['report'] = item[1][1]

        matched_dict[key1]['human'] = human_sample_json[key1]
            

        a=1
    a=1

    import cv2 # you don't need this for the search algorithm

    for human_case in matched_dict:

        human_case_ = human_case.replace('0906_fair_face_clean_','').replace('0825_fair_face_clean_','').replace('.png','.jpg')
        
        human_path = '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/images/'+ human_case_

        human_label = matched_dict[human_case]['human']
        human_title = ''
        for attr in human_label:
            row_text = attr+': '
            for item in human_label[attr]:
                row_text += item + ' '+ str(human_label[attr][item]) + ' '
            row_text += '\n'
            human_title += row_text
            

        out_paths, out_title = [human_path] , [human_title]

        matched_dict_assets = matched_dict[human_case]
        for asset_case in matched_dict_assets:
            if asset_case == 'human': continue
            asset_path = '/Users/minghaoliu/Desktop/HITL_navi/data/asset/images/'+matched_dict_assets[asset_case]['name']

            cur_report = matched_dict_assets[asset_case]['report']
            cur_title = ''
            for item in cur_report:
                if item =='top_direction':
                    if sum(cur_report[item]) != 0:
                        cur_title += item + ' ' + str(round(sum(cur_report[item]),1))  + ' \n'
                else:
                    if item == 'total': continue
                    if cur_report[item] != 0:
                        cur_title += item + ' ' + str(round(cur_report[item],1)) + ' \n'

            # cur_title = ''
            # cur_title += 'td'+str(round(matched_dict_assets[asset_case]['report']['top_direction'][0],1)) + '  '
            # cur_title += 'td'+str(round(matched_dict_assets[asset_case]['report']['top_direction'][1],1)) + '\n'
            # cur_title += 'tl'+str(round(matched_dict_assets[asset_case]['report']['top_length'],1)) + '  '
            # cur_title += 'tc'+str(round(matched_dict_assets[asset_case]['report']['top_curly'],1)) + '\n'
            # cur_title += 'sl'+str(round(matched_dict_assets[asset_case]['report']['side_length'],1)) + '  '
            # cur_title += 'sc'+str(round(matched_dict_assets[asset_case]['report']['side_curly'],1)) + '\n'
            

            cur_title += 'score '+str(round(matched_dict_assets[asset_case]['score'],1))
            out_title += [cur_title] # TODO: add report also
            out_paths.append(asset_path)
            print(human_path,asset_path)
            a=1

        concat_img = data_utils.concat_list_image(out_paths,out_title)
        # save concat_img
        cv2.imwrite('tune_search/'+human_case+'.jpg',concat_img)


    
    # print(algo.eval_distance(human,asset))