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
            'top_direction': 3, 
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
        distance_dict = {'total':0}
        for attr in human:
            human_attris = [ data_utils.attr2int(attr) for attr in human[attr]]
            asset_attris = [ data_utils.attr2int(attr) for attr in asset[attr]]

            
            if attr in self.distance_score_quality:
                distance_ = self.distance_score_quality[attr] * type_distance(human_attris,asset_attris)# Type distance: Type error when no intersection
            elif attr in self.distance_score_linear:
                mean_human, mean_asset = sum(human_attris)/len(human_attris), sum(asset_attris)/len(asset_attris)
                distance_  = self.distance_score_linear[attr] * abs(mean_human - mean_asset) # Linear distance: absolute difference
                # distance_  = distance_score_linear[attr] * (linear_distance(human_attris, asset_attris))
            elif attr in self.distance_score_type:
                if attr == 'top_direction':    
                    human_attris_g1, human_attris_g2 = data_utils.intersection_list(human_attris,self.top_direction_group1_int), data_utils.intersection_list(human_attris,self.top_direction_group2_int)
                    asset_attris_g1, asset_attris_g2 = data_utils.intersection_list(asset_attris,self.top_direction_group1_int), data_utils.intersection_list(asset_attris,self.top_direction_group2_int)
                    distance1 = self.distance_score_type[attr] * type_distance(human_attris_g1,asset_attris_g1) # Type distance
                    distance2 = self.distance_score_type[attr] * type_distance(human_attris_g2,asset_attris_g2) # Type distance
                    distance_ = [distance1,distance2]
                else:
                    distance_ = self.distance_score_type[attr] * type_distance(human_attris,asset_attris) # Type distance: Type error when no intersection

            distance_sum = distance_ if type(distance_) != list else sum(distance_)
            # if distance_sum != 0:
            #     print(attr, distance_)
            distance_dict[attr] = distance_
            distance_dict['total'] += distance_sum

        return distance_dict, distance_dict['total']


if __name__ == '__main__':
    print('search_algo.py')
    human_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/FairFace2.0/all_results_soft.json'
    human_sample_json = json.load(open(human_sample_json_path))

    asset_sample_json_path =  '/Users/minghaoliu/Desktop/HITL_navi/data/asset/820_faceattribute_round2_asset_translate_soft.json'
    asset_sample_json = json.load(open(asset_sample_json_path))

    algo = search_algorithm()
    for key1 in human_sample_json:
        for key2 in asset_sample_json:
            distance_dict, total = algo.eval_distance(human_sample_json[key1],asset_sample_json[key2])
            print(key1,key2,total)
            print(distance_dict)
            break
        break


    
    # print(algo.eval_distance(human,asset))