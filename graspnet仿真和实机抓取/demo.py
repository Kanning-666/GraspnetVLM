""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup
import socket
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


'''
def send_to_matlab(pos, ori, host='127.0.0.1', port=9999):
    """通过 TCP 发送抓取点位数据给 MATLAB"""
    try:
        
        data = {"pos": pos, "ori": ori}
        message = json.dumps(data) + "\n"  # 注意换行符，否则 MATLAB 的 readLine() 读不全
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        s.sendall(message.encode('utf-8'))
        s.close()
        print(f"[Python] 已发送到 MATLAB: {message}")
    except Exception as e:
        print(f"[Python] 发送失败: {e}")
'''
def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def _make_coord_frame_at(position, rotation, size=0.05):
    """在给定位姿处生成坐标轴(红X,绿Y,蓝Z)"""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3,  3] = position
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    # 为前3个抓取位姿叠加坐标轴，显示xyz方向
    axis_frames = []
    topk = min(3, len(gg))
    for i in range(topk):
        pos = gg[i].translation
        rot = gg[i].rotation_matrix
        axis_frames.append(_make_coord_frame_at(pos, rot, size=0.05))

    o3d.visualization.draw_geometries([cloud, *grippers, *axis_frames])
    o3d.visualization.draw_geometries([cloud, *grippers])



def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)
    # 获取抓取坐标
    gg.nms()
    gg.sort_by_score()
    
    # 获取最佳抓取的位置和姿态
    # for i in range(30):
    best_grasp = gg[0]  # 得分最高的抓取
    position = best_grasp.translation  # 3D坐标 (x, y, z)
    rotation = best_grasp.rotation_matrix  # 3×3旋转矩阵
    score = best_grasp.score  # 得分

    r = R.from_matrix(rotation)
    quaternion = r.as_quat()  

    print(f"最佳抓取位置: {position}")
    print(f"旋转矩阵:\n{rotation}")
    print(f"得分: {score}")
    print(f"四元数: {quaternion}")

    HOST = '127.0.0.1'  # 改为另一台电脑的IP地址
    PORT = 30000
    
    # 组装为字符串消息：POS:x,y,z;QUAT:qx,qy,qz,qw\n
    msg = "POS:{:.6f},{:.6f},{:.6f};QUAT:{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
        position[0], position[1], position[2],
        quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("连接成功")
            s.sendall(msg.encode('utf-8'))
            print("信息已发送：", msg.strip())
    except Exception as e:
        print(f"连接失败：{e}")


if __name__=='__main__':
    data_dir = 'doc/example_data'

    demo(data_dir)
