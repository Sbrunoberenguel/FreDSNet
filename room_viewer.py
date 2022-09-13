import os
import cv2
import torch
import argparse
import numpy as np
import open3d as op
from tqdm import tqdm

def get_3Dpoints(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = (1.0-2*x/float(W))*np.pi
    phi = (0.5-y/float(H))*np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),(sp)])
    pts = vec * dep
    return pts.transpose([1,2,0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_dir', default='Example/',
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be horizontal.')                        
    parser.add_argument('--data', default='Results/',
                        help='Directory to depth data')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.data,'point_clouds'),exist_ok=True)
    # Prepare image to processed
    img_names = os.listdir(args.img_dir)
    img_names.sort()

    #Output resolution
    H,W = 512, 1024

    for name in tqdm(img_names):
        img = cv2.imread(os.path.join(args.img_dir,name))
        img = cv2.cvtColor(cv2.resize(img,(W,H)),cv2.COLOR_BGR2RGB)
        seg = cv2.imread(os.path.join(args.data,'semantic',name[:-4]+'_seg.png'))
        seg = cv2.cvtColor(cv2.resize(seg,(W,H),interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB)
        dep = torch.FloatTensor(np.load(os.path.join(args.data,'depth',name[:-3]+'npy'))).unsqueeze(0).unsqueeze(0)
        dep = torch.nn.functional.interpolate(dep,(H,W),mode='nearest').squeeze(0)
        dep = dep.numpy()
        pts = get_3Dpoints(dep,H,W)
        pcd = op.geometry.PointCloud()
        pcd.points = op.utility.Vector3dVector(pts.reshape(-1,3))
        pcd.colors = op.utility.Vector3dVector(img.reshape(-1,3)/255.)
        op.visualization.draw_geometries([pcd],window_name=name)
        pcd2 = op.geometry.PointCloud()
        pcd2.points = op.utility.Vector3dVector(pts.reshape(-1,3))
        pcd2.colors = op.utility.Vector3dVector(seg.reshape(-1,3)/255.)
        op.visualization.draw_geometries([pcd2],window_name=name)
        op.io.write_point_cloud(os.path.join(args.data,'point_clouds',name[:-3]+'pcd'), pcd)
        op.io.write_point_cloud(os.path.join(args.data,'point_clouds',name[:-4]+'_seg.pcd'), pcd2)
