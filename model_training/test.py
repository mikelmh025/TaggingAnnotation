

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



def param2match(index, pred,label_index_dict,source_img_paths):
    pred = pred.clone().detach().cpu().numpy()
    batch_size = pred.shape[0]
    out_dicts = {}
    for i in range(batch_size):
        out_dicts[index[i].item()] = {}
    # out_dicts = [{}]*batch_size

    for label_type in label_index_dict:
        pred_cur = pred[:,label_index_dict[label_type][0]:label_index_dict[label_type][1]+1]
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

    all_scores = {}
    all_reports = {}
    for i in range(batch_size):
        dis_scores = {}
        dis_reports = {}

        for asset_key in asset_data:
            dis_dict, dis_sum = algo.eval_distance(out_dicts[index[i].item()],asset_data[asset_key])
            dis_scores[asset_key] = dis_sum
            dis_reports[asset_key] = dis_dict
        # sort dis_scores by value
        dis_scores = dict(sorted(dis_scores.items(), key=lambda d:d[1]))
        dis_scores = {k:dis_scores[k] for k in list(dis_scores)[:args.get_top_k]}
        dis_reports = {k:dis_reports[k] for k in list(dis_scores)[:args.get_top_k]}

        all_scores[index[i].item()] = dis_scores
        all_reports[index[i].item()] = dis_reports

    for key in all_scores:
        human_path = source_img_paths[key]
        image_name  = human_path.split('/')[-1]
        matched_paths = []
        for asset_key in all_scores[key]:
            matched_paths.append(args.data_root+ 'asset/images/'+asset_key)
        out_list = [human_path]+matched_paths
        out_titles = ['']*len(out_list)

        im_concat = data_utils.concat_list_image(out_list,out_titles)
        cont_save_dir = str(args.result_dir)
        os.makedirs(cont_save_dir, exist_ok=True)
        cv2.imwrite(str(cont_save_dir+'/'+image_name), im_concat)

    return all_scores

def test(args, model, test_dataset,label_index_dict):
    print('testing...')
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    source_img_paths = test_dataset.source_img_paths
    correct = 0
    total = 0
    for i, (images, labels, index) in enumerate(test_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        outputs = model(images)
        param2match(index, outputs,label_index_dict,source_img_paths)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return

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
        model = resnet_pre(num_classes)

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