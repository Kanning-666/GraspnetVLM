%%
server = tcpserver("0.0.0.0", 30000, "ByteOrder", "little-endian");
disp('Server is now listening for incoming connections...');
%% 连接机械臂
clear all;
addpath(genpath('G:\MatlabWorkSpace\wangxin\chatGPT\demo\JACO3API'))
% 机械臂
IP_ADDRESS = '192.168.1.10';
[result, tempHandle, ~] = kortexApiMexInterface('CreateRobotApisWrapper', IP_ADDRESS, 'admin', 'admin', uint32(60000), uint32(2000));
%%
%******* Jaco3 Robot hand *********%
ports = serialportlist;
s = serialport("COM9",115200); % 根据实际调整
% initialize command
commandInitialize = "FFFEFDFC010802010000000000FB";
commandInitialize2 = "FFFEFDFC0108010100A5A5A5A5FB";
% position control
commandClose = "FFFEFDFC010602010030000000FB";
commandOpen= "FFFEFDFC01060201005F000000FB";
write(s,commandInitialize,'string');
write(s,commandInitialize2,'string');

% 爪子角度
commandJointRead = 'FFFEFDFC010702000000000000FB';
commandJointWirte = 'FFFEFDFC010702010064000000FB';
%write(s,commandJointWirte,'string');
%write(s,commandJointRead,'string');

%% 力矩控制
commandForceCatch = 'FFFEFDFC01050201001E000000FB'; % 以30N的力抓取
write(s,commandForceCatch,'string'); 
% 位置控制
commandPosOpen =    'FFFEFDFC010602010059000000FB'; % 松开爪子
write(s,commandPosOpen,'string');
commandCatch =       'FFFEFDFC010602010000000000FB';
% 读取抓取状态
commandCatchRead = 'FFFEFDFC010F01000000000000FB';
%% 机械臂末端与爪子的变换关系
T_Base_marker = [-0.9928   -0.1199    0.0044    0.0540;
    0.1200   -0.9924    0.0244    0.2049;
    0.0014    0.0248    0.9997   -0.0246;
         0         0         0    1.0000];
quat = [ 0.80327917 -0.58686332  0.0536738   0.0863316]; % 移动相机之后重新设置
quat_new = [quat(4) quat(1) quat(2) quat(3)];
T_camera_marker = eye(4);T_camera_marker(1:3,4) = [0.00247103 -0.13412746  0.56548183]';T_camera_marker(1:3,1:3) = quat2dcm(quat_new)';
T_marker_camera = T_camera_marker;T_marker_camera(1:3,1:3) = T_camera_marker(1:3,1:3)';T_marker_camera(1:3,4) = -T_camera_marker(1:3,1:3)'*T_camera_marker(1:3,4);
T_Base_camera = T_Base_marker*T_marker_camera;
T_end_hand = eye(4); T_end_hand(1:3,1:3) = rotz(30*pi/180);T_end_hand(3,4) = 0.000; % 抓取状态
T_end_hand_approach = T_end_hand;
T_end_hand_approach(3,4) = T_end_hand(3,4) + 0.05; % 抓取就位状态
T_hand_end = eye(4); T_hand_end(1:3,1:3) = T_end_hand(1:3,1:3)';T_hand_end(1:3,4) = -T_end_hand(1:3,1:3)'*T_end_hand(1:3,4);
T_hand_end_approach = eye(4); T_hand_end_approach(1:3,1:3) = T_end_hand_approach(1:3,1:3)';T_hand_end_approach(1:3,4) = -T_end_hand_approach(1:3,1:3)'*T_end_hand_approach(1:3,4); % 根据实际调整
%% 获取初始位姿 
[~, base_feedback, ~, feedback_error] = kortexApiMexInterface('RefreshFeedback', tempHandle);
init_X = base_feedback.tool_pose(1);
init_Y = base_feedback.tool_pose(2);
init_Z = base_feedback.tool_pose(3);
init_Rx = base_feedback.tool_pose(4);
init_Ry = base_feedback.tool_pose(5);
init_Rz = base_feedback.tool_pose(6);
fprintf('初始位姿: X=%.3f, Y=%.3f, Z=%.3f\n', init_X, init_Y, init_Z);
ret = base_feedback.tool_pose;
Robot_T = eye(4);Robot_T(1:3,1:3) = rotz(ret(6)*pi/180)*roty(ret(5)*pi/180)*rotx(ret(4)*pi/180);Robot_T(1:3,4) = ret(1:3)';
Robot_T
%%
disp('等待 Python 连接...');

% 等待连接（阻塞）
while server.Connected == false
    pause(0.1);
end
disp('Python 已连接');

% 从Python接收数据
data = readline(server);
disp(['接收到数据: ', data]);

% 解析为数值
grab_point = str2double(split(data, ','));
disp('解析后的抓取点:');
disp(grab_point);

grab_point_xyz = grab_point(1:3);
grab_point_R = reshape(grab_point(4:end), [3, 3]);  
grab_point_R = grab_point_R';
grab_point_R
%% 
% 这个部分需要根据Aruco标签反馈
quat = [0.175 -0.193 0.648 0.715];
quat_new = [quat(4) quat(1) quat(2) quat(3)];
Target_camera = eye(4); 
Target_camera(1:3,4) = [0.088 -0.126 0.438]';
Target_camera(1:3,1:3) = quat2dcm(quat_new)';
%Target_camera(1:3,1:3) = grab_point_R;
%Target_camera(1:3,4) = grab_point_xyz';
% 对于姿态，可以修改成符合定义的抓取状态
R_change_catch = [0,0,1;...
                   1,0,0;...
                   0,1,0];
T_change_catch = eye(4);
T_change_catch(1:3,1:3) = R_change_catch;
T_change_catch(1,4) = -0.2; %特定修改，针对标定误差
Target_camera = Target_camera*T_change_catch;

Target_robot = T_Base_camera*Target_camera; % 这是爪子的位置
Target_end = Target_robot*T_hand_end; % 最终控制位姿
Target_end_approach = Target_robot*T_hand_end_approach; % 最终控制位姿

%Target_end_approach(1:3,4) = Robot_T(1:3,4);
% SE3插值
T_traj_approach = se3_interp(Robot_T, Target_end_approach, 10, 10);
show_SE3Traj(T_traj_approach) % 检查是否有问题
% 得到最终的六维控制参数
eul_extrinsic_control_traj = zeros(size(T_traj_approach, 3),3);
position_control_traj = squeeze(T_traj_approach(1:3,4,:))'; % K*3
for i = 1:size(T_traj_approach, 3)
    eul_intrinsic_control = rotm2eul(T_traj_approach(1:3,1:3,i), 'ZYX'); % 顺序是 Z Y X
    eul_extrinsic_control_traj(i,:) = eul_intrinsic_control(end:-1:1); 
end
eul_extrinsic_control_traj = eul_extrinsic_control_traj*180/pi;

T_traj_catch = se3_interp(Target_end_approach, Target_end, 10, 1);
show_SE3Traj(T_traj_catch) % 检查是否有问题
eul_extrinsic_control_traj_catch = zeros(size(T_traj_catch, 3),3);
position_control_traj_catch = squeeze(T_traj_catch(1:3,4,:))'; % K*3
for i = 1:size(T_traj_catch, 3)
    eul_intrinsic_control = rotm2eul(T_traj_catch(1:3,1:3,i), 'ZYX'); % 顺序是 Z Y X
    eul_extrinsic_control_traj_catch(i,:) = eul_intrinsic_control(end:-1:1); 
end
eul_extrinsic_control_traj_catch = eul_extrinsic_control_traj_catch*180/pi;
%% 控制部分
for i = 1:size(T_traj_approach, 3) % 到达抓取点上方
    [result] = kortexApiMexInterface('ReachCartesianPose', tempHandle, int32(0), [0, 0], 0, ...
        position_control_traj(i,:), eul_extrinsic_control_traj(i,:)); % 控制机械臂末端
    pause(1);
end

for i = 1:size(T_traj_catch, 3) % 到达抓取点
    [result] = kortexApiMexInterface('ReachCartesianPose', tempHandle, int32(0), [0, 0], 0, ...
        position_control_traj_catch(i,:), eul_extrinsic_control_traj_catch(i,:)); % 控制机械臂末端
    pause(1);
end
pause(1);
write(s,commandCatch,'string');
pause(3);
flush(s);
write(s,commandCatchRead,'string');
CatchState = read(s,28,"char");
if CatchState(20) == '3'
    disp('抓取成功');
else
    disp('抓取失败');
end
%% 结束
[result] = kortexApiMexInterface('DestroyRobotApisWrapper', uint32(tempHandle));
clear s;
clear server