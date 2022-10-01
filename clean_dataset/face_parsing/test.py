#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi !=18: continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    # print("Number of pixel got labeled: ", index[0].shape[0], "Image name:", save_path.split('/')[-1])
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im
    return index[0].shape[0]

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth)          : os.makedirs(respth)
    if not os.path.exists(respth+'_hat')   : os.makedirs(respth+'_hat')
    if not os.path.exists(respth+'_No_hat'): os.makedirs(respth+'_No_hat')
    

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net#.cuda()
    save_pth = osp.join('res/cp', cp)
    # net.load_state_dict(torch.load(save_pth))
    net.load_state_dict(torch.load(save_pth,map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))
            if 18 in np.unique(parsing):
                pixel_count = vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=osp.join(respth, image_path))
                if pixel_count > 1000:
                    img_save = cv2.imread(osp.join(dspth, image_path))
                    cv2.imwrite(osp.join(respth+'_hat', image_path), img_save)
                    continue

            cv2.imwrite(osp.join(respth+'_No_hat', image_path), img_save)

            # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))



class Evaluator(object):
    def __init__(self,save_pth, if_cuda) -> None:
        n_classes = 19
        device = torch.device('cuda' if if_cuda else 'cpu')
        self.net = BiSeNet(n_classes=n_classes)
        if if_cuda: self.net.cuda()
        self.net.load_state_dict(torch.load(save_pth,map_location=device))
        self.net.eval()

        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    def parse(self, image_path, return_image=False):
        img = Image.open(image_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = self.transform(image)
        img = torch.unsqueeze(img, 0)
        out = self.net(img)[0]
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)

        if return_image: return parsing, image
        return parsing


if __name__ == "__main__":

    evaluate(dspth='/Users/bytedance/Desktop/data/image datasets/fairface-img-margin125-trainval/val', cp='79999_iter.pth')


