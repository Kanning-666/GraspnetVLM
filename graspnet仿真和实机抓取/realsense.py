import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from datetime import datetime
import os
import subprocess

class RGBDCamera:
    serial_counter = 0

    def __init__(self, serial_number, rgb_resolution, depth_resolution):
        self.serial_number = serial_number
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.serial_id = RGBDCamera.serial_counter
        RGBDCamera.serial_counter += 1
        self.pipeline = rs.pipeline()
        self.config = self.configure_pipeline()
        self.align = rs.align(rs.stream.color)

    def configure_pipeline(self):
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, *self.rgb_resolution, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, *self.depth_resolution, rs.format.z16, 30)
        return config

    def shoot(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print(f"No frames received from camera {self.serial_number}")
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def get_intrinsics_matrix(self):
        profile = self.pipeline.get_active_profile()
        rgb_intrinsics_raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # To a matrix
        rgb_intrinsics = np.array([[rgb_intrinsics_raw.fx, 0, rgb_intrinsics_raw.ppx],
                                   [0, rgb_intrinsics_raw.fy, rgb_intrinsics_raw.ppy],
                                   [0, 0, 1]])

        depth_intrinsics_raw = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # To a matrix
        depth_intrinsics = np.array([[depth_intrinsics_raw.fx, 0, depth_intrinsics_raw.ppx],
                                   [0, depth_intrinsics_raw.fy, depth_intrinsics_raw.ppy],
                                   [0, 0, 1]])
        
        return rgb_intrinsics, rgb_intrinsics_raw.coeffs, depth_intrinsics, depth_intrinsics_raw.coeffs
    
    def get_intrinsics_raw(self):
        profile = self.pipeline.get_active_profile()
        rgb_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return rgb_intrinsics, rgb_intrinsics.coeffs, depth_intrinsics, depth_intrinsics.coeffs
    
    def get_depth_scale(self):
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()
    
    def get_pointcloud(self, depth_trunc):
        color_image, depth_image = self.shoot()
        # get intrinsic parameters
        rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = self.get_intrinsics_matrix()
        depth_scale = self.get_depth_scale()

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1/depth_scale,
            depth_trunc=depth_trunc
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width=color_image.shape[1],
                height=color_image.shape[0],
                fx=rgb_intrinsics[0, 0],
                fy=rgb_intrinsics[1, 1],
                cx=rgb_intrinsics[0, 2],
                cy=rgb_intrinsics[1, 2]
            )
        )

        return pcd

    def start(self):
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()

    def save_images(self, save_dir="./captured_images"):
        """拍照并保存RGB图和深度图"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        color_image, depth_image = self.shoot()
        
        if color_image is None or depth_image is None:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        color_path = os.path.join(save_dir, f"color.png")
        depth_path = os.path.join(save_dir, f"depth.png")
        
        # 保存RGB图(需要转换BGR)
        cv2.imwrite(color_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        # 保存深度图
        cv2.imwrite(depth_path, depth_image)
        
        extra_dir = r"D:\studentcreate\vlm\vlm"
        os.makedirs(extra_dir, exist_ok=True)
        extra_color_path = os.path.join(extra_dir, "color.png")
        extra_depth_path = os.path.join(extra_dir, "depth.png")
        cv2.imwrite(extra_color_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(extra_depth_path, depth_image)
        # print(f"Images saved: {color_path}, {depth_path}")
        return True


def get_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    device_serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    device_serials.sort()
    return device_serials

if __name__ == "__main__":
    # 获取连接的相机设备
    devices = get_devices()
    
    if len(devices) == 0:
        print("未检测到RealSense相机!")
        exit()
    
    print(f"检测到 {len(devices)} 个相机:")
    
    # 使用第一个相机
    camera = RGBDCamera(
        serial_number=devices[0],
        rgb_resolution=(1280, 720),
        depth_resolution=(1280, 720)
    )
    
    print("\n启动相机...")
    camera.start()
    
    try:
        # 等待相机稳定
        import time
        print("相机预热中...")
        time.sleep(2)
        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(ROOT_DIR, 'doc', 'example_data')
        os.makedirs(save_dir, exist_ok=True)
        success = camera.save_images(save_dir=save_dir)
        
        if not success:
            print("拍照失败!")
            exit(1)

        print("图片保存成功!")
    
    except Exception as e:
        print(f"错误: {e}")
        exit(1)
    finally:
        print("\n关闭相机...")
        camera.stop()
        print("完成!")

        '''
        result = subprocess.run(['bash', 'command_demo.sh'], cwd=ROOT_DIR)
        if result.returncode != 0:
            print("command_demo.sh 运行失败")
            exit(result.returncode)
        print("command_demo.sh 运行完成")
        '''