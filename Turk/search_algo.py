import data_utils
import json
class search_algorithm ():
    def __init__(self):
        self.distance_score_quality= {
            'blur': 0, 
            'head_occlusion': 0, 
            'mutiperson': 0
            }

        # Consider non-linear disntace for each level
        # Consider some random number subscript for each level
        self.distance_score_linear = {
            'top_curly': 1, 
            'top_length': 1.5, 
            'side_curly': 2, 
            'side_length': 2, 
            'braid_count': 8, 
            }

        self.distance_score_type = {
            'top_direction': 0, 
            'braid_tf': 10, 
            'braid_type': 3, 
            'braid_position': 2}

        self.top_direction_group1     = ['0-向下','1-斜下','2-横向','3-斜上','6-向上（头发立起来）','7-向后梳头（长发向后梳，大背头，马尾）']
        self.top_direction_group2     = ['4-中分','5-37分']
        self.top_direction_group1_int = [ data_utils.attr2int(attr) for attr in self.top_direction_group1]
        self.top_direction_group2_int = [ data_utils.attr2int(attr) for attr in self.top_direction_group2]

    def eval_distance(self,human,asset,check=False):
        def type_distance(human,asset):
            return int(len(data_utils.intersection_list(human,asset)) <1)

        def attr_vote2_list(vote_dict):
            out_list = []
            for key in vote_dict:
                out_list += [data_utils.attr2int(key)]*vote_dict[key]
            return out_list

        # MANUAL fix: use top direction for long hair only 
        flag_long_hair = True if data_utils.most_frequent(attr_vote2_list(human['side_length']))[0] >=3 else False

        distance_dict = {'total':0}
        for attr in human:
            distance_ = 0
            human_attris = attr_vote2_list(human[attr])
            asset_attris = attr_vote2_list(asset[attr])
            
            if attr in self.distance_score_quality:
                distance_ = self.distance_score_quality[attr] * type_distance(human_attris,asset_attris)# Type distance: Type error when no intersection


            # Linear distance: aggregate by average
            elif attr in self.distance_score_linear:
                mean_human, mean_asset = sum(human_attris)/len(human_attris), sum(asset_attris)/len(asset_attris)
                distance_  = self.distance_score_linear[attr] * abs(mean_human - mean_asset) # Linear distance: absolute difference
            elif attr in self.distance_score_type:

                # Top direction: split tagges into 2 groups,  
                if attr == 'top_direction':
                    # MANUAL fix: use top direction for long hair only 
                    
                    self.distance_score_type[attr] = 0.2 if not flag_long_hair else 3

                    # Split taggs into 2 groups
                    human_attris_g1, human_attris_g2 = data_utils.intersection_list(human_attris,self.top_direction_group1_int), data_utils.intersection_list(human_attris,self.top_direction_group2_int)
                    asset_attris_g1, asset_attris_g2 = data_utils.intersection_list_keep_size(asset_attris,self.top_direction_group1_int), data_utils.intersection_list_keep_size(asset_attris,self.top_direction_group2_int)
                    
                    # only calculate the distance when human has tag in a group,
                    # aggregate human and asset tags by majority vote
                    # TODO: what if one side of the party doesn't have tag? idea 1: simply ignore this term in this case; idea 2: conside as a mistake tag
                    if len(human_attris_g1) > 0 and len(asset_attris_g1) > 0:
                        human_attris_g1, asset_attris_g1 = data_utils.most_frequent(human_attris_g1), data_utils.most_frequent(asset_attris_g1)
                        distance_ += self.distance_score_type[attr] * type_distance(human_attris_g1,asset_attris_g1) # Type distance
                    if len(human_attris_g2) > 0 and len(asset_attris_g2) > 0:
                        human_attris_g2, asset_attris_g2 = data_utils.most_frequent(human_attris_g2), data_utils.most_frequent(asset_attris_g2)
                        distance_ += self.distance_score_type[attr] * type_distance(human_attris_g2,asset_attris_g2) # Type distance

                    # Manual fix : if not tag in group 1, then consider as a mistake tag
                    if len(asset_attris_g1) == 0 and flag_long_hair:
                        distance_ += self.distance_score_type[attr] * 1
                    a=1
                else:
                    # majority vote of asset_attris
                    human_attris, asset_attris = data_utils.most_frequent(human_attris), data_utils.most_frequent(asset_attris)
                    distance_ = self.distance_score_type[attr] * type_distance(human_attris,asset_attris) # Type distance: Type error when no intersection

            distance_sum = distance_ if type(distance_) != list else sum(distance_)
            distance_dict[attr] = distance_
            distance_dict['total'] += distance_sum

        return distance_dict, distance_dict['total']


if __name__ == '__main__':
    print('search_algo.py')
    human_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/all_results_soft.json'
    human_sample_json = json.load(open(human_sample_json_path))

    asset_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/asset/820_faceattribute_round2_asset_translate_soft.json'
    asset_sample_json = json.load(open(asset_sample_json_path))
    human_limit = 50
    
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