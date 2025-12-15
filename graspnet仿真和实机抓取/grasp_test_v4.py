import os
import sys
import time
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
import math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import sim

# 相机内参
CAMERA_INTRINSICS = {
    "ppx": 320.00, 
    "ppy": 240.00, 
    "fx": 589.37, 
    "fy": 589.37
}

# 配置参数
CONFIG = {
    'checkpoint_path': './logs/log_rs/checkpoint-rs.tar',  # GraspNet模型路径
    'num_point': 20000,            # 点云采样点数
    'num_view': 300,               # 视角数量
    'collision_thresh': 0.01,      # 碰撞检测阈值
    'voxel_size': 0.01,            # 体素大小
    'depth_factor': 1.0,           # 深度因子
    'angle_threshold': 35,         # 垂直抓取角度限制（度）
    'pre_grasp_height': 0.1,       # 预抓取高度（米）
    'step_size': 0.02              # 抓取步进（米）
}


class UR5GraspSystem:
    """UR5机器人抓取系统：连接仿真、图像获取、抓取预测、执行抓取"""
    
    def __init__(self):
        """初始化系统"""
        self.clientID = -1
        self.net = None
        self.rgb_camera_handle = -1
        self.depth_camera_handle = -1
        self.target_handle = -1
        self.gripper_handle = -1
        
    def connect_simulator(self):
        """连接CoppeliaSim仿真器"""
        print("连接CoppeliaSim...")
        sim.simxFinish(-1)  # 关闭所有现有连接
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        
        if self.clientID == -1:
            print("无法连接到CoppeliaSim!")
            return False
            
        print("成功连接到CoppeliaSim")
        
        # 获取对象句柄
        res, self.rgb_camera_handle = sim.simxGetObjectHandle(
            self.clientID, 'kinect_rgb', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print("获取RGB相机句柄失败")
            return False
            
        res, self.depth_camera_handle = sim.simxGetObjectHandle(
            self.clientID, 'kinect_depth', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print("获取深度相机句柄失败")
            return False
            
        res, self.target_handle = sim.simxGetObjectHandle(
            self.clientID, 'target', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print("获取target句柄失败")
            return False
            
        return True
    
    def init_grasp_net(self):
        """初始化GraspNet网络模型"""
        print("加载GraspNet模型...")
        self.net = GraspNet(
            input_feature_dim=0,
            num_view=CONFIG['num_view'],
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        print("GraspNet模型加载完成")
    
    def get_camera_images(self):
        """从相机获取RGB图像和深度图像"""
        # 获取RGB图像
        res, resolution, rgb_data = sim.simxGetVisionSensorImage(
            self.clientID, self.rgb_camera_handle, 0, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print("获取RGB图像失败")
            return None, None
            
        # 处理RGB图像
        rgb_img = np.array(rgb_data, dtype=np.uint8)
        rgb_img.resize([resolution[1], resolution[0], 3])
        rgb_img = cv2.flip(rgb_img, 0)  # CoppeliaSim图像需要垂直翻转
        
        # 获取深度图像
        res, resolution, depth_data = sim.simxGetVisionSensorDepthBuffer(
            self.clientID, self.depth_camera_handle, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print("获取深度图像失败")
            return rgb_img, None

        # 处理深度图像
        depth_img = np.array(depth_data, dtype=np.float32)
        depth_img = depth_img.reshape(resolution[1], resolution[0]) 
        depth_img = cv2.flip(depth_img, 0)  # CoppeliaSim图像需要垂直翻转

        return rgb_img, depth_img
    
    def predict_grasp(self, rgb_img, depth_img):
        """使用GraspNet预测最佳抓取位姿"""
        if rgb_img is None or depth_img is None:
            print("无效的图像数据")
            return None
        
        h, w = depth_img.shape
        d_near = 0.01
        d_far = 3.5
        # 反归一化
        depth_real = d_near + depth_img * (d_far - d_near)
        # 计算相机内参
        fov_rad = math.radians(57.0)
        fy = (h / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        
        print(f"计算得到的相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # 创建坐标网格
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # 反投影公式
        X = (u_coords - cx) * depth_real / fx
        Y = (v_coords - cy) * depth_real / fy
        Z = depth_real
        
        # 数据处理
        color = rgb_img.astype(np.float32) / 255.0  # 归一化到0-1
        
        # 创建掩码
        h, w = depth_img.shape
        workspace_mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 3
        y_coords, x_coords = np.ogrid[:h, :w]
        mask_circle = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
        workspace_mask[mask_circle] = 255
        
        factor_depth = 1000.0 
        camera = CameraInfo(
            width=w, height=h,
            fx=CAMERA_INTRINSICS['fx'], fy=CAMERA_INTRINSICS['fy'],
            cx=CAMERA_INTRINSICS['ppx'], cy=CAMERA_INTRINSICS['ppy'],
            scale=factor_depth
        )
        
        # 组合成点云
        cloud = np.stack([X, Y, Z], axis=-1)  # shape: (h, w, 3)
        depth = (depth_img * 1000).astype(np.uint16)  # 转换为毫米
        mask = (workspace_mask > 0) & (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        # 点云采样
        if len(cloud_masked) >= CONFIG['num_point']:
            idxs = np.random.choice(len(cloud_masked), CONFIG['num_point'], replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), CONFIG['num_point'] - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        # 网络推理
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
        
        end_points = dict()
        end_points['point_clouds'] = cloud_tensor
        end_points['cloud_colors'] = color_sampled
        
        # 得到位姿
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        
        if len(gg) == 0:
            print("网络没有预测出任何抓取姿态")
            return None
        
        # 抓取过滤
        if CONFIG['collision_thresh'] > 0:
            mfcdetector = ModelFreeCollisionDetector(cloud_masked, voxel_size=CONFIG['voxel_size'])
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=CONFIG['collision_thresh'])
            gg = gg[~collision_mask]
        
        if len(gg) == 0:
            print("碰撞检测后没有剩余抓取")
            return None
        
        # 分数排序
        gg.nms()
        gg.sort_by_score()
        
        return gg[0]  # 最佳抓取

    
    def transform_camera_to_world(self, grasp):
        """使用pose_handler的坐标转换方法"""
        res, camera_position = sim.simxGetObjectPosition(
            self.clientID, self.depth_camera_handle, -1, sim.simx_opmode_blocking)
        res, camera_orientation = sim.simxGetObjectOrientation(
            self.clientID, self.depth_camera_handle, -1, sim.simx_opmode_blocking)
        
        print(f"相机位置: {camera_position}")
        print(f"相机方向: {camera_orientation}")
        
        # 构建相机变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = camera_position
        r = R.from_euler('zyx', camera_orientation[::-1], degrees=False)
        rotation_matrix = r.as_matrix()
        transform_matrix[:3, :3] = rotation_matrix
        
        translation = grasp.translation
        rotation = grasp.rotation_matrix
        print(f"抓取旋转矩阵: \n{rotation}")
        
        point = np.array([translation[0], translation[1], translation[2]])
        rotation_cam = rotation
        
        # 计算最终的坐标框架
        dest_frame = np.hstack((rotation_cam, point.reshape(3, 1)))
        dest_frame = np.vstack((dest_frame, [0, 0, 0, 1]))
        
        transformed_frame = transform_matrix @ dest_frame
        print(f"最终坐标转换结果: {transformed_frame[:3, 3]}")
        
        return transformed_frame
    
    def control_gripper(self, is_open):
        """控制夹爪开合"""
        signal_value = 1 if is_open else 0
        sim.simxSetIntegerSignal(self.clientID, 'RG2_open', signal_value, sim.simx_opmode_oneshot)
    
    def execute_grasp(self, grasp_transform):
        """执行抓取动作"""
        if grasp_transform is None:
            print("无效的抓取变换")
            return False
        
        # 提取位置和旋转
        position = grasp_transform[:3, 3]
        rotation_matrix = grasp_transform[:3, :3]
        rotation_euler = R.from_matrix(rotation_matrix).as_euler('xyz')
        
        print("打开夹爪")
        self.control_gripper(True)
        time.sleep(1)
        
        # 移动到预抓取位置
        pre_grasp_position = position.copy()
        pre_grasp_position[2] += CONFIG['pre_grasp_height']
        print(f"移动到预抓取位置: {pre_grasp_position}")
        sim.simxSetObjectPosition(self.clientID, self.target_handle, -1, 
                               pre_grasp_position, sim.simx_opmode_oneshot)
        sim.simxSetObjectOrientation(self.clientID, self.target_handle, -1, 
                                  rotation_euler, sim.simx_opmode_oneshot)
        time.sleep(2)
        
        # 逐步下降到抓取位置
        steps = int(CONFIG['pre_grasp_height'] / CONFIG['step_size']) + 1
        for i in range(steps):
            current_z = pre_grasp_position[2] - i * CONFIG['step_size']
            if current_z <= position[2]:
                current_z = position[2]
            
            current_position = pre_grasp_position.copy()
            current_position[2] = current_z
            
            sim.simxSetObjectPosition(self.clientID, self.target_handle, -1, 
                                   current_position, sim.simx_opmode_oneshot)
            time.sleep(0.5)
            
            if current_z == position[2]:
                break

        # 额外下压 2cm
        extra_down = 0.02
        extra_position = position.copy()
        extra_position[2] = max(position[2] - extra_down, 0.0)
        sim.simxSetObjectPosition(self.clientID, self.target_handle, -1,
                               extra_position, sim.simx_opmode_oneshot)
        time.sleep(0.5)
        
        print("关闭夹爪抓取物体")
        self.control_gripper(False)
        time.sleep(1.5)
        
        print("抬起物体")
        sim.simxSetObjectPosition(self.clientID, self.target_handle, -1, 
                               pre_grasp_position, sim.simx_opmode_oneshot)
        time.sleep(2)
        
        return True
    
    def run(self):
        """运行完整的抓取流程"""
        try:
            # 连接仿真器
            if not self.connect_simulator():
                return False
            
            # 初始化抓取网络
            self.init_grasp_net()
            
            while True:
                print("\n===== 开始新一轮抓取 =====")
                
                rgb_img, depth_img = self.get_camera_images()
                if rgb_img is None or depth_img is None:
                    print("获取图像失败，重试中...")
                    time.sleep(1)
                    continue
                
                cv2.imwrite('debug_rgb.png', rgb_img)
                cv2.imwrite('debug_depth.png', depth_img * 5000)  
                
                # 预测抓取姿态
                best_grasp = self.predict_grasp(rgb_img, depth_img)
                if best_grasp is None:
                    print("抓取预测失败，重试中...")
                    time.sleep(1)
                    continue
                
                # 打印最佳抓取信息
                print(f"最佳抓取位置: {best_grasp.translation}")
                print(f"抓取分数: {best_grasp.score:.4f}")
                
                # 坐标转换
                grasp_transform = self.transform_camera_to_world(best_grasp)
                print(f"世界坐标系中的抓取位置: {grasp_transform[:3, 3]}")
                
                user_input = input("执行这个抓取? (y/n): ")
                if user_input.lower() != 'y':
                    print("跳过本次抓取")
                    continue
                
                # 执行抓取
                self.execute_grasp(grasp_transform)
                
                # 询问是否继续
                user_input = input("继续下一轮抓取? (y/n): ")
                if user_input.lower() != 'y':
                    break
            
            return True
            
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            print("关闭仿真连接...")
            sim.simxFinish(self.clientID)


if __name__ == "__main__":
    grasp_system = UR5GraspSystem()
    grasp_system.run()