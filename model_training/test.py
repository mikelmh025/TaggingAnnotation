

# -*- coding:utf-8 -*-
import enum
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from data.datasets import input_dataset, input_dataset_face_attr, input_dataset_face_attr_test
from models import *
import argparse
import numpy as np
from search_algo  import search_algorithm 
import json
import data_utils
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.05)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--loss', type = str, help = 'ce, gce, dmi, flc, uspl,spl,peerloss', default = 'ce')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'test')

parser.add_argument('--dataset', type = str, help = ' cifar10 or fakenews', default = 'face_attribute')
parser.add_argument('--model', type = str, help = 'cnn,resnet,vgg', default = 'resnet')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=8, help='how many subprocesses to use for data loading')
parser.add_argument('--device', type=str, help='cuda or cpu ', default='cuda')
parser.add_argument('--data_root', type=str, help='path to dataset', default='/media/otter/navi_hitl/')
parser.add_argument('--human_dataset', type=str, help='path to dataset', default='FairFace2.0/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--checkpoint', type=str, help='path to checkpoint', default='debug_dir/tagging/resnet/msebest.pt')
parser.add_argument('--get_top_k', type=int, default=1, help='top k matched images in output')
parser.add_argument('--target_mode', type=str, help='use tag or img(direct) to train', default='tag')
parser.add_argument('--otherSystem', type=str, help='use tag or img(direct) to train', default=None)

args = parser.parse_args()

# Golbal variables of attributes
continuous_attr_dict = {
    'top_curly':4,
    'top_length':6,
    'side_curly':4,
    'side_length':5
}
discrete_attr_dict = {
    'braid_tf':2,
    'braid_type':5,
    'braid_count':4,
    'braid_position':3
}
multi_class_attr_dict = {
    'top_direction':8
}
#Search algorithm
algo = search_algorithm()

if args.otherSystem is not None:
    asset_json_path = '/home/minghao/Documents/taggingAnnotation/data/' + args.otherSystem +'/soft_label.json'
    asset_img_root = '/home/minghao/Documents/taggingAnnotation/data/'+args.otherSystem+'/image/'
else:
    asset_json_path = args.data_root + 'asset/soft_label.json'
    asset_img_root = args.data_root + 'asset/image/'

with open(asset_json_path, 'r') as f:
    asset_data = json.load(f)

# Sort dict by key,return a dict
def sort_dict(data):
    return dict(sorted(data.items(), key=lambda d:d[0]))
asset_data = sort_dict(asset_data)

# given tensor pred or original labsl, return a dict of tag (used for search algorithm)
def tensor2tagDict(input_tensor, index, label_index_dict):
    input_tensor = input_tensor.clone().detach().cpu().numpy()
    batch_size = input_tensor.shape[0]

    out_dicts = {}
    for i in range(batch_size):
        out_dicts[index[i].item()] = {}

    for label_type in label_index_dict:
        pred_cur = input_tensor[:,label_index_dict[label_type][0]:label_index_dict[label_type][1]+1]
        # label_cur = label[:,label_index_dict[label_type][0]:label_index_dict[label_type][1]+1]
        if label_type in continuous_attr_dict:
            for i in range(batch_size):
                out_dicts[index[i].item()][label_type] = {str(max(pred_cur[i][0],0)):1}

        elif label_type in discrete_attr_dict:
            for i in range(batch_size):
                out_dicts[index[i].item()][label_type] = {str(np.argmax(pred_cur[i])):1}

        elif label_type in multi_class_attr_dict:
            for i in range(batch_size):
                out_dicts[index[i].item()][label_type] = {str(np.argmax(pred_cur[i])):1}

    return out_dicts

# Given dict of tag search assets
def tag_search_asset(tag_dict,index):
    batch_size =len(tag_dict)
    # New version: Get top K matched assets
    all_search_scores = {}
    all_search_reports = {}
    for i in range(batch_size):
        search_scores, search_reports = algo.multi_round_search(tag_dict[index[i].item()],asset_data)
        all_search_scores[index[i].item()] = search_scores
        all_search_reports[index[i].item()] = search_reports
    return all_search_scores, all_search_reports

def search_report2img(all_search_scores,all_search_reports, source_img_paths, idx_key, args=None):
    # Get basic info of human matched assets
    human_path = source_img_paths[idx_key]
    image_name  = human_path.split('/')[-1]
    matched_assets = list(all_search_scores[idx_key].keys())[:args.get_top_k]

    # Convert to image path
    # matched_paths = [os.path.join(args.data_root,'asset/images/',asset_name) for asset_name in matched_assets]
    matched_paths = [os.path.join(asset_img_root,asset_name) for asset_name in matched_assets]

    
    out_list = [human_path]+matched_paths

    # Add titles
    # out_titles = ['']*len(out_list)
    out_titles = ['']
    
    for matched_paths in matched_paths:
        match_namme = matched_paths.split('/')[-1]
        report_ = all_search_reports[idx_key][match_namme]
        title_ = [attr + ' ' + str(round(report_[attr],2)) + ' \n' for attr in report_ if report_[attr] != 0 ] 
        title_ = ''.join(title_)
        out_titles.append(title_)

    im_concat = data_utils.concat_list_image(out_list,out_titles)
    return im_concat, image_name


def param2match(labels, pred,extra_info_dict):
    data_dict = {}
    batch_size = labels.shape[0]

    index = extra_info_dict['index']
    label_index_dict = extra_info_dict['label_index_dict']
    source_img_paths = extra_info_dict['source_img_paths']
    all_asset_names = extra_info_dict['all_asset_names']
    extra_label = extra_info_dict['extra_label']
    args = extra_info_dict['args']
    asset_label_dict = extra_info_dict['asset_label_dict']
    

    if args.target_mode == 'tag':
        pred_tag_dict = tensor2tagDict(pred, index,label_index_dict)
        label_tag_dict = tensor2tagDict(labels, index,label_index_dict)
        
        # Search asset
        label_search_scores, label_search_reports = tag_search_asset(label_tag_dict,index)
        pred_search_scores, pred_search_reports = tag_search_asset(pred_tag_dict,index)
        

        # Calculate accuracy / matching chance of pred and label
        for key in pred_search_scores:
            target = list(label_search_scores[key].keys())[0] # option1

            pred_top1_score, pred_top5_score = list(pred_search_scores[key].values())[0], list(pred_search_scores[key].values())[0:5]
            label_top1_score, label_top5_score = list(label_search_scores[key].values())[0], list(label_search_scores[key].values())[0:5]

            pred_top1_dist, pred_top5_dist = pred_top1_score, sum(pred_top5_score)/len(pred_top5_score)
            label_top1_dist, label_top5_dist = label_top1_score, sum(label_top5_score)/len(label_top5_score)
            

            pred_assets = list(pred_search_scores[key].keys())

            # Top1 strict
            pred_top1_acc_strict = 1 if target in pred_assets[0]  else 0
            # Top5 strict
            pred_top5_acc_strict = 1 if target in pred_assets[:5] else 0
                

            data_dict['pred_top1_acc_strict'] = data_dict['pred_top1_acc_strict']+pred_top1_acc_strict if 'pred_top1_acc_strict' in data_dict else pred_top1_acc_strict
            data_dict['pred_top5_acc_strict'] = data_dict['pred_top5_acc_strict']+pred_top5_acc_strict if 'pred_top5_acc_strict' in data_dict else pred_top5_acc_strict

            data_dict['pred_top1_dist'] =  data_dict['pred_top1_dist'] + pred_top1_dist if 'pred_top1_dist' in data_dict else pred_top1_dist
            data_dict['pred_top5_dist'] =  data_dict['pred_top5_dist'] + pred_top5_dist if 'pred_top5_dist' in data_dict else pred_top5_dist
            data_dict['label_top1_dist'] =  data_dict['label_top1_dist'] + label_top1_dist if 'label_top1_dist' in data_dict else label_top1_dist
            data_dict['label_top5_dist'] =  data_dict['label_top5_dist'] + label_top5_dist if 'label_top5_dist' in data_dict else label_top5_dist

        # Save images
        cont_save_dir = str(args.result_dir)
        os.makedirs(cont_save_dir, exist_ok=True)
        for key in pred_search_scores:
            im_concat, image_name = search_report2img(pred_search_scores,pred_search_reports, source_img_paths, key, args=args)
            cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)

            for i in range(args.get_top_k):
                single_match_path = os.path.join(cont_save_dir, 'top'+str(i+1))
                os.makedirs(single_match_path, exist_ok=True)
                matched_name = list(pred_search_scores[key].keys())[i]
                image_name_ = image_name.split('.')[0]
                save_name_ = image_name_+'_'+matched_name+'.jpg'
                # matched_path = os.path.join(args.data_root,'asset/images/',matched_name)
                matched_path = os.path.join(asset_img_root,matched_name)
                
                cv2.imwrite(str(single_match_path+'/'+save_name_), cv2.imread(matched_path))

        
        # Save images
        cont_save_dir = str(args.result_dir)+'_label'
        os.makedirs(cont_save_dir, exist_ok=True)
        for key in label_search_scores:
            im_concat, image_name = search_report2img(label_search_scores,label_search_reports, source_img_paths, key, args=args)
            cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)

    elif args.target_mode == 'img':
        pred = pred.clone().detach().cpu().numpy()
        batch_size = pred.shape[0]
        # change pred from np array to tensor
        pred = torch.from_numpy(pred)
        
        outputs = F.softmax(pred, dim=1)
        _, pred = torch.max(outputs.data, 1)
        
        # get top 5 predictions from outputs
        pred_top5 = torch.topk(outputs, 5, dim=1)[1]


        # Calculate accuracy
        # one hot label to index
        labels = labels.clone().detach().cpu().numpy()
        labels = np.argmax(labels, axis=1)
        labels = torch.from_numpy(labels)
        correct = (pred == labels).sum().item()

        # Calculate top5 accuracy
        correct_top5 = sum([i if labels[i] in pred_top5[i] else 0 for i in range(batch_size)])
        
        data_dict['pred_top1_acc_strict'] = correct
        data_dict['pred_top5_acc_strict'] = correct_top5

        human_tags = tensor2tagDict(extra_label, index,label_index_dict)
        for idx, key in enumerate(human_tags):
            outputs_ = outputs[idx]
            human_tag = human_tags[key]

            # get top1 prediction
            pred_top1_ = asset_label_dict[all_asset_names[pred[idx].item()]+'.png']
            pred_top1_dict_ = {'0':pred_top1_}

            # get top5 predictions
            pred_top5_ = torch.topk(outputs_, 5, dim=0)[1]  
            pred_top5_dict_ = {str(idx):asset_label_dict[all_asset_names[item]+'.png'] for idx, item in enumerate(pred_top5_)}

            
            
            search_scores_1, search_reports_1 = algo.multi_round_search(human_tag,pred_top1_dict_)
            search_scores_5, search_reports_5 = algo.multi_round_search(human_tag,pred_top5_dict_)

            pred_top1_dist, pred_top5_dist = sum(list(search_scores_1.values())), sum(list(search_scores_5.values()))
            a=1
            data_dict['pred_top1_dist'] =  data_dict['pred_top1_dist'] + pred_top1_dist if 'pred_top1_dist' in data_dict else pred_top1_dist
            data_dict['pred_top5_dist'] =  data_dict['pred_top5_dist'] + pred_top5_dist if 'pred_top5_dist' in data_dict else pred_top5_dist

        # pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax =  correct, correct, correct_top5, correct_top5

        matched_paths = []
        save_dir =  str(args.result_dir)+'_'+args.target_mode
        os.makedirs(save_dir, exist_ok=True)
        for i in range(batch_size):
            # all_asset_names[pred[i].item()]
            matched_path_ = args.data_root+ 'asset/images/'+all_asset_names[pred[i].item()]
            matched_name  = matched_path_.split('/')[-1]
            matched_paths += [matched_path_]

            human_path = source_img_paths[index[i].item()]
            image_name  = human_path.split('/')[-1].split('.')[0]
            save_name_ = image_name+'_'+matched_name+'.jpg'
            cv2.imwrite(str(save_dir+'/'+save_name_), cv2.imread(matched_path_+'.png'))

            for j in range(5):
                single_match_path = os.path.join(save_dir, 'top'+str(j+1))
                os.makedirs(single_match_path, exist_ok=True)
                topk_matched_path_ = args.data_root+ 'asset/images/'+all_asset_names[pred_top5[i][j]]
                topk_matched_name  = topk_matched_path_.split('/')[-1]
                save_name_ = image_name+'_'+topk_matched_name+'.jpg'
                cv2.imwrite(str(single_match_path+'/'+save_name_), cv2.imread(topk_matched_path_+'.png'))


    return  data_dict #pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax


def test(args, model, test_dataset,label_index_dict):
    print('testing...')
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    source_img_paths = test_dataset.source_img_paths
    try:
        all_asset_names = test_dataset.all_asset_names
    except:
        all_asset_names = None

    asset_label_dict = test_dataset.asset_label_dict

    correct = 0
    total = 0
    compare_dict = {}
    # correct_counts = (0,0,0,0) # pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax
    for i, (images, labels, index,extra_label) in tqdm(enumerate(test_loader)):
        if index[0].item() >=300: break
        images = images.to(args.device)
        labels = labels.to(args.device)
        outputs = model(images)

        extra_info_dict = {
            'index': index,
            'source_img_paths': source_img_paths,
            'all_asset_names': all_asset_names,
            'label_index_dict': label_index_dict,
            'args': args,
            'extra_label': extra_label,
            'asset_label_dict': asset_label_dict
        }
        compare_dict_ = param2match(labels, outputs, extra_info_dict)
        # correct_counts = tuple(map(sum, zip(correct_counts, correct_counts_)))
        total += labels.size(0)
        compare_dict = {key:value+compare_dict[key] if key in compare_dict else value for key,value in compare_dict_.items()}
    #     _, predicted = torch.max(outputs.data, 1)
    #     correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    # pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax = correct_counts

    if 'pred_top1_acc_strict' in compare_dict:
        print('pred_top1_acc_strict: %.3f %%' % (100 * compare_dict['pred_top1_acc_strict'] / total))
    if 'pred_top5_acc_strict' in compare_dict:
        print('pred_top5_acc_strict: %.3f %%' % (100 * compare_dict['pred_top5_acc_strict'] / total))

    if 'pred_top1_dist' in compare_dict:
        print('Pred top1 dist: %.3f' % (compare_dict['pred_top1_dist'] / total))
    if 'pred_top5_dist' in compare_dict:
        print('Pred top5 dist: %.3f' % (compare_dict['pred_top5_dist'] / total))
    
    if 'label_top1_dist' in compare_dict:
        print('Label top1 dist: %.3f' % (compare_dict['label_top1_dist'] / total))
    if 'label_top5_dist' in compare_dict:
        print('Label top5 dist: %.3f' % (compare_dict['label_top5_dist'] / total))

    # print('pred_top1_acc_relax: %.3f %%' % (100 * pred_top1_acc_relax / total))
    # print('pred_top5_acc_relax: %.3f %%' % (100 * pred_top5_acc_relax / total))

    return

def main(args):
    batch_size = args.batch_size
    learning_rate = args.lr

    if args.debug: args.num_workers = 0
    # noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    # args.noise_type = noise_type_map[args.noise_type]
    # load dataset
    # train_dataset,test_dataset,num_classes,num_training_samples = input_dataset_face_attr(args, args.dataset,root=None)
    # test_dataset = test_dataset
    test_dataset,num_classes,num_training_samples = input_dataset_face_attr_test(args, args.dataset,root=None)
    label_index_dict = test_dataset.get_index_dict()

    # load model
    print('building model...')
    if args.model == 'cnn':
        model = CNN(input_channel=3, n_outputs=num_classes)
    if args.model == 'vgg':
        model = vgg11()
    elif args.model == 'inception':
        model = Inception3()
    else:
        if '34' in args.model:
            model = resnet_pre(num_classes,resnet_option='34')
        elif '50' in args.model:
            model = resnet_pre(num_classes,resnet_option='50')
        elif '101' in args.model:
            model = resnet_pre(num_classes,resnet_option='101')
        

    if torch.cuda.device_count() > 1 : # and not args.debug:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(args.device)

    # load checkpoint
    if args.checkpoint:
        print('loading checkpoint...')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint, strict=True)

    test(args, model, test_dataset,label_index_dict)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



# Past version: 10/30/2022 8:53pm

# def param2match(args, index, labels, pred,label_index_dict,source_img_paths,all_asset_names=None):
#     batch_size = labels.shape[0]

#     if args.target_mode == 'tag':
#         pred_tag_dict = tensor2tagDict(pred, index,label_index_dict)
#         label_tag_dict = tensor2tagDict(labels, index,label_index_dict)
        
#         # Search asset
#         label_search_scores, label_search_reports = tag_search_asset(label_tag_dict,index)
#         pred_search_scores, pred_search_reports = tag_search_asset(pred_tag_dict,index)
        

#         # Calculate accuracy / matching chance of pred and label
#         # Using top 1 asset / top 5 assets
#         # Option1: use min as the ground truth option, option2: use all the assets with min score as ground truth options
#         pred_top1_acc_strict, pred_top1_acc_relax = 0, 0
#         pred_top5_acc_strict, pred_top5_acc_relax = 0, 0

#         for key in pred_search_scores:
#             target = list(label_search_scores[key].keys())[0] # option1
#             # min_scores = min(list(label_search_scores[key].values()))
#             # targets = [item for item in label_search_scores[key] if label_search_scores[key][item] == min_scores] # option2
#             # min_scores = list(label_search_scores[key].values())[0] # Using the score of first one as min score. However if we want to do multi layer search need to find another way.
#             # targets = []
#             # for item in label_search_scores[key]:
#             #     if label_search_scores[key][item] == min_scores:
#             #         targets.append(item)
#             #     else:
#             #         break
            
#             pred_top1_score, pred_top5_score = [list(label_search_scores[key].values())[0]], list(label_search_scores[key].values())[0:5]
#             label_top1_score, label_top5_score = [list(label_search_scores[key].values())[0]], list(label_search_scores[key].values())[0:5]

#             pred_top1_dist, pred_top5_dist = pred_top1_score, sum(pred_top5_score)/len(pred_top5_score)
#             label_top1_dist, label_top5_dist = label_top1_score, sum(label_top5_score)/len(label_top5_score)
            

#             pred_assets = list(pred_search_scores[key].keys())

#             # Top1 strict
#             if target in pred_assets[0]: pred_top1_acc_strict += 1
                

#             # # Top1 relax
#             # if len(data_utils.intersection_list([pred_assets[0]], targets)) > 0:
#             #     pred_top1_acc_relax += 1

#             # Top5 strict
#             if target in pred_assets[:5]: pred_top5_acc_strict += 1
                
                
#             # # Top5 relax
#             # if len(data_utils.intersection_list(pred_assets[0:top_k], targets)) > 0:
#             #     pred_top5_acc_relax += 1
#             # else:
#             #     cont_save_dir = str(args.result_dir)+'_failed'
#             #     os.makedirs(cont_save_dir, exist_ok=True)
#             #     im_concat, image_name = search_report2img(pred_search_scores,pred_search_reports, source_img_paths, key, args=args)
#             #     im_concat2, image_name2 = search_report2img(label_search_scores,label_search_reports, source_img_paths, key, args=args)
#             #     # vertical concat im_concat and im_concat2
#             #     im_concat = data_utils.vertical_cat([im_concat,im_concat2])
#             #     cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)

        
#         print("Known issue in the accuracy calculation:     1. # Search algorithm make sure there are enough options to match for top 5, top 10❗️❗️")
#         # # Save images
#         # cont_save_dir = str(args.result_dir)
#         # os.makedirs(cont_save_dir, exist_ok=True)
#         # for key in pred_search_scores:
#         #     im_concat, image_name = search_report2img(pred_search_scores,pred_search_reports, source_img_paths, key, args=args)
#         #     cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)

#         #     for i in range(args.get_top_k):
#         #         single_match_path = os.path.join(cont_save_dir, 'top'+str(i+1))
#         #         os.makedirs(single_match_path, exist_ok=True)
#         #         matched_name = list(pred_search_scores[key].keys())[i]
#         #         image_name_ = image_name.split('.')[0]
#         #         save_name_ = image_name_+'_'+matched_name+'.jpg'
#         #         matched_path = os.path.join(args.data_root,'asset/images/',matched_name)
#         #         cv2.imwrite(str(single_match_path+'/'+save_name_), cv2.imread(matched_path))

        
#         # # Save images
#         # cont_save_dir = str(args.result_dir)+'_label'
#         # os.makedirs(cont_save_dir, exist_ok=True)
#         # for key in label_search_scores:
#         #     im_concat, image_name = search_report2img(label_search_scores,label_search_reports, source_img_paths, key, args=args)
#         #     cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)

#     elif args.target_mode == 'img':
#         pred = pred.clone().detach().cpu().numpy()
#         batch_size = pred.shape[0]
#         # change pred from np array to tensor
#         pred = torch.from_numpy(pred)
        
#         outputs = F.softmax(pred, dim=1)
#         _, pred = torch.max(outputs.data, 1)
        
#         # get top 5 predictions from outputs
#         pred_top5 = torch.topk(outputs, 5, dim=1)[1]


#         # Calculate accuracy
#         # one hot label to index
#         labels = labels.clone().detach().cpu().numpy()
#         labels = np.argmax(labels, axis=1)
#         labels = torch.from_numpy(labels)
#         correct = (pred == labels).sum().item()
#         acc = correct / batch_size

#         # Calculate top5 accuracy
#         correct_top5 = 0
#         for i in range(batch_size):
#             if labels[i] in pred_top5[i]:
#                 correct_top5 += 1
#         acc_top5 = correct_top5 / batch_size


#         pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax =  correct, correct, correct_top5, correct_top5
#         a=1

#         # matched_paths = []
#         # save_dir =  str(args.result_dir)+'_'+args.target_mode
#         # os.makedirs(save_dir, exist_ok=True)
#         # for i in range(batch_size):
#         #     # all_asset_names[pred[i].item()]
#         #     matched_path_ = args.data_root+ 'asset/images/'+all_asset_names[pred[i].item()]
#         #     matched_name  = matched_path_.split('/')[-1]
#         #     matched_paths += [matched_path_]

#         #     human_path = source_img_paths[index[i].item()]
#         #     image_name  = human_path.split('/')[-1].split('.')[0]
#         #     save_name_ = image_name+'_'+matched_name+'.jpg'
#         #     cv2.imwrite(str(save_dir+'/'+save_name_), cv2.imread(matched_path_+'.png'))


#     return pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax