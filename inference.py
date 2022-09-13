import os
import cv2
import time
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import FreDSNet_model as model

warnings.filterwarnings('ignore')

color_code = [ [0,0,0],        #UNK
               [100,0,0],      #beam
               [0,0,100],      #board
               [255,0,0],      #bookcase
               [123,123,255],  #ceiling
               [255,123,123],  #chair
               [200,200,200],  #clutter
               [0,100,0],      #column
               [100,220,100],  #door
               [123,255,123],  #floor
               [0,0,255],      #sofa
               [0,255,0],      #table
               [50,30,100],    #wall
               [200,200,220]]  #window


def color_segmentation(seg):
    H,W = seg.shape
    cseg = seg.reshape(-1,1)
    out = np.zeros((H*W,3))
    for i in range(H*W):
        out[i] = color_code[int(cseg[i])]
    return out.reshape(H,W,3)

def decode(img,d_max):
    img = img*255 if img.max() < 1.1 else img
    R,G,B = img[:,0],img[:,1],img[:,2]
    int1 = d_max/255.0
    int2 = (d_max/255.0)/255.0
    d1 = (R*d_max)/255.0
    d2 = (G/255.0)*int1
    d3 = (B/255.0)*int2
    return d1+d2+d3

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default='ckpt/FreDSNet_ICRA_weights.pth',
                        help='path to load saved checkpoint.')
    parser.add_argument('--root_dir', required=False, default='Example')
    parser.add_argument('--out_dir',  required=False, default='Results')
    parser.add_argument('--no_depth',    required=False, action='store_true',default=False)
    parser.add_argument('--no_semantic', required=False, action='store_true',default=False)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    #PARSER END#

    device = torch.device('cpu' if args.no_cuda else 'cuda')
    print('Inference made with: {}\n'.format(device))

    net,state_dict = model.load_weigths(args)
    # net.param_count_sections()
    net.to(device)

    num_classes = net.num_classes
    scale = 2

    print('Results for FreDSNet')
    net.eval()

    img_list = os.listdir(args.root_dir)
    
    # Inferencing   
    accum_time = 0
    os.makedirs(args.out_dir,exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'semantic'),exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'depth'),exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'depthmap'),exist_ok=True)

    for name in tqdm(img_list):
        img_path = os.path.join(args.root_dir,name)

        H, W = 512//scale,1024//scale
        img = cv2.resize(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB),(W,H),cv2.INTER_CUBIC)
        img = np.array(img,np.float32)[...,:3] / 255.
        i_img_mask = np.logical_and(img[...,0]==0,img[...,1]==0,img[...,2]==0)*1
        img_mask = np.ones_like(i_img_mask) - i_img_mask
        x_img = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        x = x_img.unsqueeze(0)
        with torch.no_grad():
            t_start = time.time()
            output = net(x.to(device))    
            t_end = time.time()
        inf_time = (t_end - t_start)
        depth = output['Depth']
        pred_depth = depth.cpu().numpy().astype(np.float32).squeeze(0).squeeze(0)
        semantic = output['Semantic'].cpu().squeeze(0)
        accum_time += inf_time
           
        #Output management
        pred_sem = torch.argmax(semantic,dim=0).numpy()
        pred_sem = color_segmentation(pred_sem) + 0.25*img*255.

        #Save coded data
        cv2.imwrite(os.path.join(args.out_dir,'semantic',name[:-4]+'_seg.png'),pred_sem*img_mask.reshape(H,W,1))
        np.save(os.path.join(args.out_dir,'depth',name[:-4]+'.npy'),pred_depth*img_mask)
        plt.figure(0)
        plt.imshow(pred_depth*img_mask)
        plt.savefig(os.path.join(args.out_dir,'depthmap',name[:-4]+'_dep.png'))

    print('Total inference time: %.2f' %accum_time)
    print('Frames per second at 256 x 512 : %.2f' %(len(img_list)/accum_time))
