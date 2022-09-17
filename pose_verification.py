'''
written by Yuhang Ming on Feb 28, 2022.

The file is used to verify if pose tranformation used is correct.
It reads in a sequence of RGB or RGB-D images and their corresponding poses
and outputs visualisation of transformed joint point cloud.

The files requires open3d, opencv-python, imageio, tqdm packages to run.

ToDo:
- Visualise the camera frustums.
'''

import os
import json
import glob
import cv2
import imageio
from tqdm import tqdm, trange
import numpy as np
import open3d as o3d

def load_blender(basedir):
    # load meta file
    split = 'train'
    with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
        meta = json.load(fp)
    depth_scale = 1000.
    # load files
    imgs = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        imgs.append(fname)
        # print(fname)
        # # img_cv = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # # print(img_cv.shape)     # BGR (800, 800, 3), (800, 800, 4) with IMREAD_UNCHANGED
        # img = imageio.imread(fname)
        # # print(img.shape)        # RBG (800, 800, 4)
        # cv2.imshow('display', img)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

        poses.append(np.array(frame['transform_matrix']))
        
    H, W = imageio.imread(imgs[0]).shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    return imgs, None, poses, K, depth_scale

def load_7scenes(basedir):
    # get intrincs and sequences
    K = np.loadtxt(os.path.join(basedir, 'camera-intrinsics.txt'))
    # get the sequences
    with open(basedir+'/TestSplitMini.txt', 'r') as f:
        seqs = f.read().split('\n')[:-1]
    depth_scale = 1000.
    colors = []
    depths = []
    poses = []
    for seq in seqs:
        # get image & pose files in the directory
        content = os.listdir(os.path.join(basedir, seq))
        for file_name in content:
            if 'color' in file_name:
                colors.append(os.path.join(basedir, seq, file_name))
            # colors.sort()
            if 'depth' in file_name:
                depths.append(os.path.join(basedir, seq, file_name))
            if 'pose' in file_name:
                poses.append(os.path.join(basedir, seq, file_name))
            # poses.sort()
        colors.sort()
        depths.sort()
        poses.sort()

    return colors, depths, poses, K, depth_scale

def load_scannet(basedir):
    with open(os.path.join(basedir, basedir[-12:]+'.txt')) as f:
        for line in f.readlines():
            if 'numColorFrames' in line:
                num = int(line.strip().split(' ')[-1])
    depth_scale = 1000.
    colors = []
    depths = []
    poses = []
    for i in range(num):
        colors.append(os.path.join(basedir, 'color', str(i)+'.jpg'))
        depths.append(os.path.join(basedir, 'depth', str(i)+'.png'))
        poses.append(os.path.join(basedir, 'pose', str(i)+'.txt'))
    
    K = np.loadtxt(os.path.join(basedir, 'intrinsic/intrinsic_depth.txt'))[:3, :3]
    return colors, depths, poses, K, depth_scale

def load_replica(basedir):
    with open(basedir+'/cam_params.json') as f:
        intrinsic_data = json.load(f)['camera']
    # print(intrinsic_data)
    W = intrinsic_data['w']                 # 1200
    H = intrinsic_data['h']                 # 680
    fx = intrinsic_data['fx']               # 600.0  # focal length x
    fy = intrinsic_data['fy']               # 600.0  # focal length y
    cx = intrinsic_data['cx']               # 599.5  # optical center x
    cy = intrinsic_data['cy']               # 339.5  # optical center y
    depth_scale = intrinsic_data['scale']   # 6553.5
    K = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
    )

    # rgb and depth path
    imgs = sorted(glob.glob(f'{basedir}/frames/frame*'))
    deps = sorted(glob.glob(f'{basedir}/frames/depth*'))

    # poses
    poses = []
    with open(basedir+'/traj.txt', "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        poses.append(c2w)
    # poses = np.stack(poses, 0)
    return imgs, deps, poses, K, depth_scale

def back_project(H, W, K, dep=None, convention='OpenCV'):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    if convention == 'OpenCV':
        pcd = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
    elif convention == 'OpenGL':
        pcd = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    else:
        raise ValueError('Unsupport backproject convention:', convention)
    # print(pcd.shape)
    if dep is not None:
        if dep.ndim == pcd.ndim:
            pcd = pcd * dep
        else:
            for i in range(pcd.shape[2]):
                pcd[..., i] = pcd[..., i] * dep
            # pcd = np.multiply(pcd, dep)

    # (H, W, 3) where 3 is for x, y, z.
    return pcd


if __name__ == '__main__':
    # dataset setup
    dataset = 'replica'
    convention = 'OpenCV'
    spacing = 15

    # opencv to opengl pose transformation
    T0 = np.array(
        [[1.,  0., 0., 0.],
         [0.,  0., 1., 0.],
         [0., -1., 0., 0.],
         [0.,  0., 0., 1.]]
    )   # rotate -90 degrees about x-axis (switch axis)
    T1 = np.array(
        [[1.,  0.,  0., 0.],
         [0., -1.,  0., 0.],
         [0.,  0., -1., 0.],
         [0.,  0.,  0., 1.]]
    )   # change positive direction of y-axis and z-axis

    if dataset == '7scenes':
        imgs, depths, poses, K, depth_scale = load_7scenes('/home/yohann/NNs/data/7scenes/chess')
    elif dataset == 'scannet':
        imgs, depths, poses, K, depth_scale = load_scannet('/home/yohann/NNs/data/ScanNet/scans/scene0005_00')
    elif dataset == 'blender':
        imgs, depths, poses, K, depth_scale = load_blender('/home/yohann/NNs/data/nerf_example_data/nerf_synthetic/lego')
    elif dataset == 'replica':
        imgs, depths, poses, K, depth_scale = load_replica('/media/yohann/fastStorage/data/Nice-SLAM-data/Replica/office4/seq00')
    print("dataset information", '('+dataset+', '+convention+'):')
    print('# of RGB images:', len(imgs))
    print('# of Depth maps:', len(depths)) if depths is not None else print('No depth map found.')
    print('# of valid Poses:', len(poses))
    print('Shape of intrins:', K.shape)

    num = len(imgs)
    step = num // spacing
    rgbs = []
    pcds = []
    # for i in trange(780, 797, 15):
    for i in trange(865, 1000, 30):
    # for i in range(0, num, step):
        # get color
        img = imageio.imread(imgs[i])
        img = img[..., :3].astype(np.float32) / 255.
        
        # get depth and pose
        if depths is not None:
            dep = imageio.imread(depths[i])
            dep = dep.astype(np.float32) / depth_scale
            # dep = dep.astype(np.float32) * depth_scale / 255.
            # dep = np.repeat(dep[..., np.newaxis], 3, axis=2)
            H, W = dep.shape[:2]
            print(dep[int(H/2), 0:10])
            print(dep[int(H/2), int(W/2)-5:int(W/2)+5])
            print(dep[int(H/2), W-10:W])

            if dep.shape[0]!=img.shape[0] or dep.shape[1]!=img.shape[1]:
                img = cv2.resize(img, (dep.shape[1], dep.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)   # cv2.INTER_AREA
            
            print(poses[i])
            Rt = poses[i] if not isinstance(poses[i], str) else np.loadtxt(poses[i])
            if convention == 'OpenGL':
                # Both transformation generates consistent output
                # but the resulting coordinate systems are different
                Rt = T0 @ Rt @ T1
                # Rt[1:3, 0] *= -1
                # Rt[0, 1:3] *= -1
                # Rt[1:3,-1] *= -1
            
        else:
            dep = None
            Rt = poses[i]
        
        # back projection
        pcd = back_project(img.shape[0], img.shape[1], K, dep, convention=convention).reshape(-1, 3)

        # coordinate transformation
        # p' = R * p + t
        pcd = np.sum(pcd[..., np.newaxis, :] * Rt[:3, :3], -1) + \
                np.broadcast_to(Rt[:3, 3], np.shape(pcd))
        
        # concatenate
        rgbs.append(img.reshape(-1, 3))
        pcds.append(pcd)

    # concatenate for visualisation
    rgbs = np.concatenate(rgbs, 0)
    pcds = np.concatenate(pcds, 0)
    
    ### Visualisation
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcds)
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgbs)
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window(window_name=convention, width=640, height=480, left=0, top=480)
    o3d_vis.add_geometry(o3d_pcd)
    
    while True:
        o3d_vis.update_geometry(o3d_pcd)
        if not o3d_vis.poll_events():
            break
        o3d_vis.update_renderer()

    o3d_vis.destroy_window()

