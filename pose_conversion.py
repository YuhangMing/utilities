from distutils import filelist
from fileinput import filelineno
import os
# import json
# import glob
# import cv2
# import imageio
# from tqdm import tqdm, trange
import math
import numpy as np

def quad_to_rot(Q):
    """
    Q: qx i + qy j + qz k + qw
    """
    # Extract the values from Q
    qx = Q[0]
    qy = Q[1]
    qz = Q[2]
    qw = Q[3]
     
    # First row of the rotation matrix
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
     
    # Second row of the rotation matrix
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qw * qx)
     
    # Third row of the rotation matrix
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def rot_to_quad(rot):

    tr = rot[0,0] + rot[1,1] + rot[2,2]
    print(tr)
    if tr > 0:
        qw = math.sqrt(1+tr)/2.
        qx = (rot[1,2] - rot[2,1]) / (4. * qw)
        qy = (rot[2,0] - rot[0,2]) / (4. * qw)
        qz = (rot[0,1] - rot[1,0]) / (4. * qw)
    elif (rot[0, 0] > rot[1,1]) &(rot[0,0] > rot[2,2]):
        S = math.sqrt(1.0 + rot[0,0] - rot[1,1] - rot[2,2]) * 2; # S=4*qx 
        qw = (rot[2,1] - rot[1,2]) / S
        qx = 0.25 * S
        qy = (rot[0,1] + rot[1,0]) / S 
        qz = (rot[0,2] + rot[2,0]) / S 
    elif (rot[1,1] > rot[2,2]) : 
        S = math.sqrt(1.0 + rot[1,1] - rot[0,0] - rot[2,2]) * 2 # S=4*qy
        qw = (rot[0,2] - rot[2,0]) / S
        qx = (rot[0,1] + rot[1,0]) / S 
        qy = 0.25 * S
        qz = (rot[1,2] + rot[2,1]) / S 
    else: 
        S = math.sqrt(1.0 + rot[2,2] - rot[0,0] - rot[1,1]) * 2 # S=4*qz
        qw = (rot[1,0] - rot[0,1]) / S
        qx = (rot[0,2] + rot[2,0]) / S
        qy = (rot[1,2] + rot[2,1]) / S
        qz = 0.25 * S

    return (qx, qy, qz, qw)


if __name__ == '__main__':
    # data_format = 'tum'
    # pose_path = '/media/yohann/fastStorage/logs-nerf-slam-replica/tum_fr1_desk'
    # pose_path = '/media/yohann/fastStorage/logs-nerf-slam-replica/tum_fr2_xyz'
    pose_path = '/media/yohann/fastStorage/logs-nerf-final/tum_fr3_office'
    # pose_path = '/media/yohann/fastStorage/logs-nerf-slam-replica/scene0169_00_init_7'
    # pose_path = '/media/yohann/fastStorage/logs-nerf-slam-replica/replica_office4_final'
    # pose_path = '/media/yohann/fastStorage/logs-nerf-final/replica_room2'
    file_list = [ 'trajectory_gt.txt', 'trajectory_est_intrim.txt', 'trajectory_opt_intrim.txt']


    # # # test 
    # # data_path = '/media/yohann/fastStorage/data/Nice-SLAM-data/TUM_RGBD/rgbd_dataset_freiburg1_desk'
    # # with open(os.path.join(data_path, 'groundtruth.txt'), 'r') as f:
    # #     lines = f.readlines()
    # # for line in lines:
    # #     if line[0] == '#':
    # #         continue
    # #     val = line.strip().split(' ')
    # #     print(val)
    # #     trans = np.array([
    # #         float(val[1]),
    # #         float(val[2]),
    # #         float(val[3])
    # #     ])
    # #     quad = np.array([
    # #         float(val[4]),
    # #         float(val[5]),
    # #         float(val[6]),
    # #         float(val[7])
    # #     ])
    # #     rot = quad_to_rot(quad)
    # #     quad2 = rot_to_quad(rot)
    # #     print(quad2)
        

    # load poses from the files
    # tum
    # data_path = '/media/yohann/fastStorage/data/Nice-SLAM-data/TUM_RGBD/rgbd_dataset_freiburg1_desk'
    # data_path = '/media/yohann/fastStorage/data/Nice-SLAM-data/TUM_RGBD/rgbd_dataset_freiburg2_xyz'
    data_path = '/media/yohann/fastStorage/data/Nice-SLAM-data/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household'
    file_list = file_list[2:]
    for file in file_list:
        poses = np.loadtxt(os.path.join(pose_path, file))
        # print(poses.shape)
        # with open(os.path.join(path, file), 'r') as f:
        #     lines = f.readlines()
        # print(len(lines))

        # TUM only
        with open(os.path.join(data_path, 'associate.txt'), 'r') as f:
            lines = f.readlines()
        print(len(lines))
        timestamps = []
        for line in lines:
            timestamps.append(line.strip().split(' ')[0])
        # print(timestamps)

        # loop through the poses
        output_file = 'tum_'+file
        print(output_file)
        outF = open(os.path.join(pose_path, output_file), 'w')
        # outF.write('testaaaa')

        print(poses.shape)
        for i in range(0, poses.shape[0], 3):
            print(i//3)
            # print(poses[i:i+3, :].shape)
            # print(np.array([[0., 0., 0., 1.]]).shape)
            T = np.concatenate(
                (poses[i:i+3, :], np.array([[0., 0., 0., 1.]])), axis=0
            )
            quad = rot_to_quad(T[:3, :3])
            # print(timestamps[i//3], T[0,3], T[1,3], T[2,3], quad)
            # print(T)
            # print(i, T[:,3], quad)

            out_string = timestamps[i//3] + f' {T[0,3]:.4f}' + f' {T[1,3]:.4f}' + f' {T[2,3]:.4f}' \
                 + f' {quad[0]:.4f}' + f' {quad[1]:.4f}' + f' {quad[2]:.4f}' + f' {quad[3]:.4f}\n'
            outF.write(out_string)
            # print(out_string)
        outF.close()


    # # load poses from the files
    # # scannet kitti
    # for file in file_list:
    #     poses = np.loadtxt(os.path.join(pose_path, file))

    #     # loop through the poses
    #     output_file = 'kitti_'+file
    #     print(output_file)
        
    #     kitti_poses = []
    #     for i in range(0, poses.shape[0], 3):
    #         # print(poses[i:i+3, :].shape)
    #         # print(np.array([[0., 0., 0., 1.]]).shape)
            
    #         T = poses[i:i+3, :]
    #         # quad = rot_to_quad(T[:3, :3])
    #         # print(timestamps[i//3], T[0,3], T[1,3], T[2,3], quad)
    #         # print(T)
    #         # print(T)
    #         # print(T.reshape(1, -1))
    #         kitti_poses.append(T.reshape(1, -1))
    #     kitti_poses = np.concatenate(kitti_poses, 0)
    #     np.savetxt(os.path.join(pose_path, output_file), kitti_poses)

    # # print(file_list)