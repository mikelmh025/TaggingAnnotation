

# -*- coding:utf-8 -*-
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

asset_json_path = args.data_root + 'asset/soft_label.json'
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
    matched_paths = [os.path.join(args.data_root,'asset/images/',asset_name) for asset_name in matched_assets]
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

# TODO change Tag label and retrain
def param2match(args, index, labels, pred,label_index_dict,source_img_paths,all_asset_names=None):
    batch_size = labels.shape[0]

    if args.target_mode == 'tag':
        pred_tag_dict = tensor2tagDict(pred, index,label_index_dict)
        label_tag_dict = tensor2tagDict(labels, index,label_index_dict)
        
        # Search asset
        pred_search_scores, pred_search_reports = tag_search_asset(pred_tag_dict,index)
        label_search_scores, label_search_reports = tag_search_asset(label_tag_dict,index)

        # Calculate accuracy / matching chance of pred and label
        # Using top 1 asset / top 5 assets
        # Option1: use min as the ground truth option, option2: use all the assets with min score as ground truth options
        pred_top1_acc_strict, pred_top1_acc_relax = 0, 0
        pred_top5_acc_strict, pred_top5_acc_relax = 0, 0

        for key in pred_search_scores:
            target = list(label_search_scores[key].keys())[0] # option1
            min_scores = min(list(label_search_scores[key].values()))
            targets = [item for item in label_search_scores[key] if label_search_scores[key][item] == min_scores] # option2

            pred = list(pred_search_scores[key].keys())

            # Top1 strict
            if target in pred[0]:
                pred_top1_acc_strict += 1
            # Top1 relax
            if len(list(set(pred[0]).intersection(set(targets)))) > 0:
                pred_top1_acc_relax += 1

            # Top5 strict
            if target in pred[:5]:
                pred_top5_acc_strict += 1
            if len(list(set(pred[0:5]).intersection(set(targets)))) > 0:
                pred_top5_acc_relax += 1

        # Save images
        cont_save_dir = str(args.result_dir)
        os.makedirs(cont_save_dir, exist_ok=True)
        for key in pred_search_scores:
            im_concat, image_name = search_report2img(pred_search_scores,pred_search_reports, source_img_paths, key, args=args)
            cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)

        
        # Save images
        cont_save_dir = str(args.result_dir)+'_label'
        os.makedirs(cont_save_dir, exist_ok=True)
        for key in label_search_scores:
            im_concat, image_name = search_report2img(label_search_scores,label_search_reports, source_img_paths, key, args=args)
            cv2.imwrite(os.path.join(cont_save_dir,image_name), im_concat)


        # for key in all_scores:
        #     human_path = source_img_paths[key]
        #     image_name  = human_path.split('/')[-1]
        #     matched_paths = []
        #     for asset_key in all_scores[key]:
        #         matched_paths.append(args.data_root+ 'asset/images/'+asset_key)
        #     out_list = [human_path]+matched_paths
        #     out_titles = ['']*len(out_list)

        #     im_concat = data_utils.concat_list_image(out_list,out_titles)
        #     cont_save_dir = str(args.result_dir)
        #     os.makedirs(cont_save_dir, exist_ok=True)
        #     cv2.imwrite(str(cont_save_dir+'/'+image_name), im_concat)

        #     for i in range(args.get_top_k):
        #         single_match_path = os.path.join(cont_save_dir, 'top'+str(i+1))
        #         os.makedirs(single_match_path, exist_ok=True)
        #         matched_name = matched_paths[i].split('/')[-1].split('.')[0]
        #         image_name_ = image_name.split('.')[0]
        #         save_name_ = image_name_+'_'+matched_name+'.jpg'
        #         cv2.imwrite(str(single_match_path+'/'+save_name_), cv2.imread(matched_paths[i]))
    elif args.target_mode == 'img':
        pred = pred.clone().detach().cpu().numpy()
        batch_size = pred.shape[0]
        # change pred from np array to tensor
        pred = torch.from_numpy(pred)
        
        outputs = F.softmax(pred, dim=1)
        _, pred = torch.max(outputs.data, 1)

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


    return pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax


def test(args, model, test_dataset,label_index_dict):
    print('testing...')
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    source_img_paths = test_dataset.source_img_paths
    try:
        all_asset_names = test_dataset.all_asset_names
    except:
        all_asset_names = None

    correct = 0
    total = 0

    correct_counts = (0,0,0,0) # pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax
    for i, (images, labels, index) in enumerate(test_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        outputs = model(images)
        correct_counts_ = param2match(args, index, labels, outputs,label_index_dict,source_img_paths,all_asset_names)
        correct_counts = tuple(map(sum, zip(correct_counts, correct_counts_)))
        total += labels.size(0)
    #     _, predicted = torch.max(outputs.data, 1)
    #     correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    pred_top1_acc_strict, pred_top1_acc_relax, pred_top5_acc_strict, pred_top5_acc_relax = correct_counts
    print('pred_top1_acc_strict: %.3f %%' % (100 * pred_top1_acc_strict / total))
    print('pred_top1_acc_relax: %.3f %%' % (100 * pred_top1_acc_relax / total))
    print('pred_top5_acc_strict: %.3f %%' % (100 * pred_top5_acc_strict / total))
    print('pred_top5_acc_relax: %.3f %%' % (100 * pred_top5_acc_relax / total))

    return

def main(args):
    batch_size = args.batch_size
    learning_rate = args.lr

    if args.debug: args.num_workers = 0
    # noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    # args.noise_type = noise_type_map[args.noise_type]
    # load dataset
    # train_dataset,test_dataset,num_classes,num_training_samples = input_dataset_face_attr(args, args.dataset,root=None,human_dir='v3')
    test_dataset,num_classes,num_training_samples = input_dataset_face_attr_test(args, args.dataset,root=None,human_dir='v3')
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