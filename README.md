本文档主要讲解如何运行整个项目
# step1 Windows复现graspnet
见github网址[graspnet/graspnet-baseline: Baseline model for "GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping" (CVPR 2020)](https://github.com/graspnet/graspnet-baseline)
先在本地复现graspnet

---
### Requirements
- python 3.9
- cuda 11.8
- pytorch 
- Open3d >=0.8
- TensorBoard 2.3
- NumPy
- SciPy
- Pillow
- tqdm
---

注意：由于之后实机使用的是深度相机realsense d435i，所以graspnet的权重要选择rs的权重，如果使用kinect再用kn的权重。
本机复现全过程以及具体遇到问题和解决请见`Windows复现graspnet及问题解决.pdf`

# step2 仿真运行
- 首先安装coppeliasim4.1.0
- 仿真文件存在graspnet-baseline\coppeliasim文件夹中，打开vrep_lab3.ttt仿真文件，运行仿真场景
- 运行graspnet-baseline\grasp_test_v4.py

# step3 实机运行
1. 前提是标定好相机和机械臂

2. 使用深度相机获取图像，得到的color.png和depth.png会放到`.\doc\example_data`和`.\vlm`中（注意更改文件中的存放地址），也放到vlm文件夹中
```
python realsense.py
```

3. 在当前目录下创建文件夹vlm进入，创建一个虚拟环境本地部署qwen
*注意：transformers>=4.49.0才能和qwen2.5-VL版本匹配*

---
### Requirements
- python 3.11
- cuda 11.8
- torch 2.4.0+cu118
- torchvision 0.19.0+cu118
- modelscope
- transformers 4.57.0
- numpy              1.26.4
- openai-whisper 
- opencv-python  
- pillow  12.0.0
- pydub  0.25.1
- scipy  1.16.3
- sounddevice  0.5.3
- ultralytics 8.3.230
---

qwen2.5-VL本地部署过程如下：
```bash
 # 1.虚拟环境中下载模型文件
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir 你的本地文件地址

# 2.安装所需要的库：
# pytorch可以从https://download.pytorch.org/whl/torch/网站上下载cuda对应版本的torch，从https://download.pytorch.org/whl/torchvision/下载对应版本的torchvision，再pip install
pip install numpy requests pillow scipy pydub
pip install sounddevice
# 需额外装ffmpeg
# Whisper 依赖 ffmpeg：
# 下载 ffmpeg 并添加到系统环境变量 添加bin的路径
pip install openai-whispe 
pip install opencv-python  
pip install ultralytics
pip install transformers ( pip install transformers==4.57.0 )

# (可选)3.加速模型处理必要的库：
pip install accelerate
pip install qwen-vl-utils
pip install 'vllm>0.7.2'
```

4. 在此虚拟环境python=3.11中运行vlm_gai_2.py（注：语音识别时去讯飞开放平台获取自己的api）
```
python vlm_gai_2.py
```
当出现录音字样时就说出指令（如“识别并切割图中的白色长方体”）
等待最终得到的掩码图存放在`graspnet-baseline\doc\example_data`中（注意更改文件中的存放地址）

5. 在step1的虚拟环境中运行graspnet模型
```
sh command_demo.sh
```

6. 在控制实机的主机上MATLAB运行catch_controller2.m
将graspnet计算出的最佳抓取位姿传给catch_controller2.m中

# step4 制作数据集
拍摄多角度不同物体不同复杂度的环境下的数据，进行step3的“语音识别-视觉切割-获取掩码”全流程，得到colorx.png，depthx.png，maskx.png（x为正整数排序），比对图片后验证了模型的准确性。