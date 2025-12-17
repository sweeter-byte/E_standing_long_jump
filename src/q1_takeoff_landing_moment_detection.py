"""
问题1：起跳/落地时刻检测 + 滞空阶段运动描述
============================================
解决以下问题：
1. Y轴方向验证 (向下为正)
2. 异常值检测：使用统计方法(IQR/Z-score)，同时处理X和Y
3. 有效范围检测：多指标融合，可视化验证
4. 起跳/落地检测：状态机+多指标，处理左右脚不对称
5. 质心计算：加权平均所有关键点
6. 可视化：原始vs平滑对比，不确定性分析

坐标系说明：
- 图像坐标系：Y轴向下为正
- 站立时：鼻子Y < 髋部Y < 脚踝Y
- 跳起时：Y值减小；落地时：Y值增大
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 配置 
OUTPUT_DIR = Path("../output/q1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR = "../data/attachments/attachment1/"

# 关键点索引
KP = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
}

# 身体部位质量比例（基于生物力学文献）
# 参考: Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement
BODY_MASS_RATIO = {
    'head': 0.081,      # 头部 (点0-10)
    'trunk': 0.497,     # 躯干 (点11-12, 23-24)
    'upper_arm': 0.028, # 上臂 (点13-14) x2
    'forearm': 0.016,   # 前臂 (点15-16) x2
    'hand': 0.006,      # 手 (点17-22) x2
    'thigh': 0.100,     # 大腿 (点25-26) x2
    'shank': 0.047,     # 小腿 (点27-28) x2
    'foot': 0.014,      # 脚 (点29-32) x2
}

# 默认帧率
FPS = 30


# 数据加载
def load_data(filepath):
    """加载xlsx数据"""
    df = pd.read_excel(filepath)
    frames = df['帧号'].values
    
    coord_cols = [col for col in df.columns if col != '帧号']
    coords = df[coord_cols].values
    keypoints = coords.reshape(len(frames), 33, 2)
    
    return frames, keypoints


def print_data_statistics(keypoints, name=""):
    """打印数据统计信息，帮助调试"""
    print(f"\n[数据统计] {name}")
    print(f"  形状: {keypoints.shape}")
    print(f"  X范围: [{np.nanmin(keypoints[:,:,0]):.1f}, {np.nanmax(keypoints[:,:,0]):.1f}]")
    print(f"  Y范围: [{np.nanmin(keypoints[:,:,1]):.1f}, {np.nanmax(keypoints[:,:,1]):.1f}]")
    
    # 检查异常值（0或极端值）
    zero_count = np.sum(keypoints == 0)
    nan_count = np.sum(np.isnan(keypoints))
    print(f"  零值数量: {zero_count}, NaN数量: {nan_count}")
    
    # 关键点统计
    ankle_y = (keypoints[:, KP['left_ankle'], 1] + keypoints[:, KP['right_ankle'], 1]) / 2
    print(f"  脚踝Y: mean={np.nanmean(ankle_y):.1f}, std={np.nanstd(ankle_y):.1f}")


# 异常值检测与处理
def detect_outliers_iqr(data, k=1.5):
    """使用IQR方法检测异常值"""
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (data < lower) | (data > upper)


def detect_outliers_zscore(data, threshold=3):
    """使用Z-score方法检测异常值"""
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std < 1e-6:
        return np.zeros(len(data), dtype=bool)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold


def detect_velocity_outliers(data, threshold=3):
    """基于速度突变检测异常值"""
    velocity = np.diff(data, prepend=data[0])
    return detect_outliers_zscore(velocity, threshold)


def interpolate_outliers(data, outlier_mask):
    """使用线性插值修复异常值（更稳定）"""
    if not np.any(outlier_mask):
        return data.copy()
    
    valid_idx = np.where(~outlier_mask)[0]
    if len(valid_idx) < 2:
        # 太少有效点，用中位数填充
        return np.full_like(data, np.nanmedian(data))
    
    # 使用线性插值（比样条更稳定，避免外推问题）
    result = np.interp(np.arange(len(data)), valid_idx, data[valid_idx])
    
    # 边界处理：确保不超出有效数据范围
    data_min = np.min(data[valid_idx])
    data_max = np.max(data[valid_idx])
    result = np.clip(result, data_min, data_max)
    
    return result


def preprocess_keypoints(keypoints, verbose=True):
    """
    预处理关键点数据
    - 使用统计方法检测异常值（同时处理X和Y）
    - 使用样条插值修复
    - 自适应平滑滤波
    """
    kp_clean = keypoints.copy().astype(float)
    n_frames, n_kp, _ = keypoints.shape
    
    total_outliers = 0
    
    for kp_idx in range(n_kp):
        for coord in range(2):
            series = kp_clean[:, kp_idx, coord]
            
            # 方法1: 检测零值和极小值（跟踪丢失）
            zero_mask = series < 1
            
            # 方法2: IQR异常检测
            iqr_mask = detect_outliers_iqr(series, k=2.0)
            
            # 方法3: 速度突变检测
            vel_mask = detect_velocity_outliers(series, threshold=4)
            
            # 综合判断：任一方法检测到即为异常
            outlier_mask = zero_mask | iqr_mask | vel_mask
            
            if np.any(outlier_mask):
                total_outliers += np.sum(outlier_mask)
                series[outlier_mask] = np.nan
                # 使用样条插值修复
                valid_idx = np.where(~np.isnan(series))[0]
                if len(valid_idx) > 0:
                    series = interpolate_outliers(
                        np.nan_to_num(series, nan=np.nanmean(series)), 
                        np.isnan(series)
                    )
                kp_clean[:, kp_idx, coord] = series
    
    if verbose:
        print(f"  异常值处理: 检测并修复 {total_outliers} 个异常点")
    
    # 自适应平滑：根据帧数调整窗口大小
    window = min(11, n_frames // 3)
    if window % 2 == 0:
        window += 1
    window = max(5, window)  # 至少5
    
    kp_smooth = np.zeros_like(kp_clean)
    for kp_idx in range(n_kp):
        for coord in range(2):
            if n_frames > window:
                kp_smooth[:, kp_idx, coord] = savgol_filter(
                    kp_clean[:, kp_idx, coord], window, min(3, window-2)
                )
            else:
                kp_smooth[:, kp_idx, coord] = kp_clean[:, kp_idx, coord]
    
    if verbose:
        print(f"  平滑滤波: window={window}, polyorder={min(3, window-2)}")
    
    return kp_smooth, kp_clean  # 返回平滑和清洗后的原始数据


# 有效范围检测
def find_valid_range(keypoints, verbose=True):
    """
    检测有效数据范围（排除往回走的部分）
    
    多指标融合：
    1. 质心X坐标最大点
    2. 水平速度持续为负
    3. 脚部Y坐标稳定（落地后）
    """
    # 质心X坐标
    com_x = (keypoints[:, KP['left_hip'], 0] + keypoints[:, KP['right_hip'], 0]) / 2
    
    # 脚部Y坐标
    foot_y = (keypoints[:, KP['left_heel'], 1] + keypoints[:, KP['right_heel'], 1]) / 2
    
    # 平滑
    com_x_smooth = savgol_filter(com_x, min(15, len(com_x)//2*2+1), 3)
    foot_y_smooth = savgol_filter(foot_y, min(15, len(foot_y)//2*2+1), 3)
    
    # 计算速度
    vel_x = np.gradient(com_x_smooth)
    vel_y = np.gradient(foot_y_smooth)
    
    # 指标1: X坐标最大点
    max_x_frame = np.argmax(com_x_smooth)
    
    # 指标2: 从最大点开始，检测速度持续为负
    # 使用自适应阈值（基于速度标准差）
    vel_std = np.std(vel_x[:max_x_frame]) if max_x_frame > 10 else np.std(vel_x)
    vel_threshold = -0.3 * vel_std  # 自适应阈值
    
    valid_end = max_x_frame
    consecutive_negative = 0
    required_consecutive = 5
    
    for i in range(max_x_frame, min(len(vel_x) - 1, max_x_frame + 50)):
        if vel_x[i] < vel_threshold:
            consecutive_negative += 1
            if consecutive_negative >= required_consecutive:
                valid_end = i - required_consecutive + 1
                break
        else:
            consecutive_negative = 0
    
    # 指标3: 验证脚部Y已稳定（落地）
    # 检查valid_end附近脚部Y的稳定性
    if valid_end > 10:
        foot_y_var = np.var(foot_y_smooth[max(0, valid_end-10):valid_end])
        if verbose:
            print(f"  落地稳定性检查: foot_y_var={foot_y_var:.2f}")
    
    valid_end = min(valid_end, len(keypoints) - 1)
    
    if verbose:
        print(f"  X最大帧: {max_x_frame}")
        print(f"  有效范围: 0 ~ {valid_end}")
        print(f"  排除帧数: {len(keypoints) - valid_end - 1}")
    
    return valid_end, {
        'max_x_frame': max_x_frame,
        'com_x': com_x_smooth,
        'vel_x': vel_x,
        'vel_threshold': vel_threshold,
    }


# 质心计算（加权）
def calculate_weighted_com(keypoints):
    """
    使用加权平均计算质心
    权重基于身体部位质量比例（生物力学文献）
    """
    n_frames = len(keypoints)
    com = np.zeros((n_frames, 2))
    
    # 定义各部位代表点和权重
    body_parts = {
        # 头部 (8.1%)
        'head': ([0], 0.081),
        # 躯干 (49.7%) - 用肩和髋的中点
        'trunk': ([11, 12, 23, 24], 0.497),
        # 上臂 (2.8% x 2)
        'left_upper_arm': ([13], 0.028),
        'right_upper_arm': ([14], 0.028),
        # 前臂+手 (2.2% x 2)
        'left_forearm': ([15], 0.022),
        'right_forearm': ([16], 0.022),
        # 大腿 (10% x 2)
        'left_thigh': ([25], 0.100),
        'right_thigh': ([26], 0.100),
        # 小腿 (4.7% x 2)
        'left_shank': ([27], 0.047),
        'right_shank': ([28], 0.047),
        # 脚 (1.4% x 2)
        'left_foot': ([29, 31], 0.014),
        'right_foot': ([30, 32], 0.014),
    }
    
    total_weight = 0
    for part_name, (indices, weight) in body_parts.items():
        part_pos = np.mean(keypoints[:, indices, :], axis=1)  # 该部位中心
        com += part_pos * weight
        total_weight += weight
    
    com /= total_weight  # 归一化
    
    return com


def calculate_simple_com(keypoints):
    """简化质心：髋部中点（作为对比）"""
    return (keypoints[:, KP['left_hip']] + keypoints[:, KP['right_hip']]) / 2


# 起跳/落地检测（状态机）
def get_all_foot_y(keypoints):
    """获取所有脚部点的Y坐标"""
    foot_indices = [
        KP['left_ankle'], KP['right_ankle'],
        KP['left_heel'], KP['right_heel'],
        KP['left_foot_index'], KP['right_foot_index']
    ]
    return keypoints[:, foot_indices, 1]


def estimate_ground_level(keypoints):
    """估计地面Y坐标（使用脚部Y的高分位数）"""
    all_foot_y = get_all_foot_y(keypoints)
    # 取95%分位数作为地面估计
    ground_y = np.percentile(all_foot_y, 95)
    return ground_y


def detect_jump_phases_stateful(keypoints, fps=30, verbose=True):
    """
    使用状态机检测跳跃阶段
    
    状态: STANDING -> TAKEOFF -> FLIGHT -> LANDING -> LANDED
    
    多指标融合：
    1. 脚部Y坐标（所有6个脚部点）
    2. 垂直速度
    3. 地面距离
    """
    n_frames = len(keypoints)
    
    # 获取所有脚部Y
    all_foot_y = get_all_foot_y(keypoints)  # shape: (n_frames, 6)
    
    # 使用平均值和最低点（最大Y）
    foot_y_mean = np.mean(all_foot_y, axis=1)
    foot_y_lowest = np.max(all_foot_y, axis=1)  # Y向下为正，最大=最低
    
    # 平滑
    window = min(11, n_frames // 3)
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    
    foot_y_smooth = savgol_filter(foot_y_mean, window, 3)
    foot_lowest_smooth = savgol_filter(foot_y_lowest, window, 3)
    
    # 计算速度和加速度
    velocity = np.gradient(foot_y_smooth) * fps
    acceleration = np.gradient(velocity) * fps
    
    # 估计地面和站立基准
    ground_y = estimate_ground_level(keypoints)
    
    # 使用前10%帧估计站立基准
    n_baseline = max(10, n_frames // 10)
    baseline_y = np.median(foot_y_smooth[:n_baseline])
    baseline_std = np.std(foot_y_smooth[:n_baseline])
    
    if verbose:
        print(f"  地面估计: {ground_y:.1f}")
        print(f"  站立基准: {baseline_y:.1f} ± {baseline_std:.1f}")
    
    # 自适应阈值
    height_threshold = max(15, 3 * baseline_std)  # 高度阈值
    vel_threshold = np.percentile(np.abs(velocity), 90) * 0.2  # 速度阈值
    
    if verbose:
        print(f"  高度阈值: {height_threshold:.1f} pixels")
        print(f"  速度阈值: {vel_threshold:.1f} pixels/s")
    
    # 状态机检测
    # 找到Y值最小点（最高点）
    flight_peak = np.argmin(foot_y_smooth)
    
    # 从最高点向前找起跳点
    takeoff = flight_peak
    for i in range(flight_peak, -1, -1):
        # 条件：Y接近baseline 且 速度接近0
        if foot_y_smooth[i] > baseline_y - height_threshold:
            if np.abs(velocity[i]) < vel_threshold * 2:
                takeoff = i
                break
    
    # 从最高点向后找落地点
    landing = flight_peak
    for i in range(flight_peak, n_frames):
        # 条件：Y接近baseline 且 速度接近0
        if foot_y_smooth[i] > baseline_y - height_threshold:
            if velocity[i] > -vel_threshold:  # 不再下降
                landing = i
                break
    
    # 验证检测结果
    flight_duration = (landing - takeoff + 1) / fps
    if flight_duration < 0.1 or flight_duration > 2.0:
        if verbose:
            print(f"  ⚠ 警告: 滞空时间异常 ({flight_duration:.3f}s)，使用备用方法")
        # 备用方法：速度过零点
        takeoff, landing = detect_jump_phases_velocity(
            foot_y_smooth, velocity, baseline_y, height_threshold
        )
    
    # 计算不确定性（±帧数）
    uncertainty_frames = max(1, int(fps * 0.033))  # 约1帧
    
    detection_info = {
        'foot_y_raw': foot_y_mean,
        'foot_y_smooth': foot_y_smooth,
        'foot_lowest_smooth': foot_lowest_smooth,
        'velocity': velocity,
        'acceleration': acceleration,
        'ground_y': ground_y,
        'baseline_y': baseline_y,
        'baseline_std': baseline_std,
        'height_threshold': height_threshold,
        'vel_threshold': vel_threshold,
        'flight_peak': flight_peak,
        'uncertainty_frames': uncertainty_frames,
    }
    
    return takeoff, landing, detection_info


def detect_jump_phases_velocity(foot_y, velocity, baseline_y, threshold):
    """备用方法：基于速度检测"""
    min_vel_idx = np.argmin(velocity)
    
    # 起跳：速度首次显著变负
    takeoff = min_vel_idx
    vel_thresh = velocity[min_vel_idx] * 0.2
    for i in range(min_vel_idx, -1, -1):
        if velocity[i] > vel_thresh:
            takeoff = i
            break
    
    # 落地：速度恢复到接近0
    landing = min_vel_idx
    for i in range(min_vel_idx, len(velocity)):
        if velocity[i] > -abs(vel_thresh):
            landing = i
            break
    
    return takeoff, landing


# 运动分析 
def calculate_joint_angle(p1, p2, p3):
    """计算三点夹角（p2为顶点）"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    if v1.ndim == 1:
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    else:
        dot = np.sum(v1 * v2, axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    
    cos_angle = np.clip(dot / (norm + 1e-8), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def analyze_flight_phase(keypoints, takeoff, landing, fps=30):
    """分析滞空阶段"""
    flight_kp = keypoints[takeoff:landing+1]
    n_flight = len(flight_kp)
    
    analysis = {
        'n_frames': n_flight,
        'flight_time': n_flight / fps,
        'flight_time_uncertainty': 1 / fps,  # ±1帧的不确定性
    }
    
    # 质心轨迹（加权和简化）
    com_weighted = calculate_weighted_com(flight_kp)
    com_simple = calculate_simple_com(flight_kp)
    
    analysis['com'] = {
        'weighted': {'x': com_weighted[:, 0], 'y': com_weighted[:, 1]},
        'simple': {'x': com_simple[:, 0], 'y': com_simple[:, 1]},
        'x_displacement': com_weighted[-1, 0] - com_weighted[0, 0],
        'y_displacement': com_weighted[-1, 1] - com_weighted[0, 1],
    }
    
    # 最高点
    peak_idx = np.argmin(com_weighted[:, 1])
    analysis['peak'] = {
        'frame': takeoff + peak_idx,
        'relative_frame': peak_idx,
        'time': peak_idx / fps,
    }
    
    # 初始速度估算（使用抛物线拟合）
    if n_flight >= 5:
        t = np.arange(n_flight) / fps
        y = -com_weighted[:, 1]  # 反转Y，向上为正
        x = com_weighted[:, 0]
        
        # 拟合 y = y0 + v0y*t - 0.5*g*t^2
        # 简化：用前几帧估算
        dt = 1 / fps
        v0_x = (x[min(3, n_flight-1)] - x[0]) / (min(3, n_flight-1) * dt)
        v0_y = (y[min(3, n_flight-1)] - y[0]) / (min(3, n_flight-1) * dt) + 0.5 * 9.8 * (min(3, n_flight-1) * dt)
        
        analysis['initial_velocity'] = {
            'vx': v0_x,
            'vy': v0_y,  # 向上为正
            'magnitude': np.sqrt(v0_x**2 + v0_y**2),
            'angle': np.degrees(np.arctan2(v0_y, v0_x)),
        }
    
    # 关节角度
    angles = {}
    
    # 膝关节
    angles['left_knee'] = calculate_joint_angle(
        flight_kp[:, KP['left_hip']],
        flight_kp[:, KP['left_knee']],
        flight_kp[:, KP['left_ankle']]
    )
    angles['right_knee'] = calculate_joint_angle(
        flight_kp[:, KP['right_hip']],
        flight_kp[:, KP['right_knee']],
        flight_kp[:, KP['right_ankle']]
    )
    
    # 髋关节
    angles['left_hip'] = calculate_joint_angle(
        flight_kp[:, KP['left_shoulder']],
        flight_kp[:, KP['left_hip']],
        flight_kp[:, KP['left_knee']]
    )
    angles['right_hip'] = calculate_joint_angle(
        flight_kp[:, KP['right_shoulder']],
        flight_kp[:, KP['right_hip']],
        flight_kp[:, KP['right_knee']]
    )
    
    # 踝关节
    angles['left_ankle'] = calculate_joint_angle(
        flight_kp[:, KP['left_knee']],
        flight_kp[:, KP['left_ankle']],
        flight_kp[:, KP['left_foot_index']]
    )
    angles['right_ankle'] = calculate_joint_angle(
        flight_kp[:, KP['right_knee']],
        flight_kp[:, KP['right_ankle']],
        flight_kp[:, KP['right_foot_index']]
    )
    
    # 躯干角度
    shoulder_mid = (flight_kp[:, KP['left_shoulder']] + flight_kp[:, KP['right_shoulder']]) / 2
    hip_mid = (flight_kp[:, KP['left_hip']] + flight_kp[:, KP['right_hip']]) / 2
    trunk_vec = shoulder_mid - hip_mid
    angles['trunk'] = np.degrees(np.arctan2(trunk_vec[:, 0], -trunk_vec[:, 1]))
    
    analysis['angles'] = angles
    
    # 左右对称性分析
    knee_diff = np.abs(angles['left_knee'] - angles['right_knee'])
    hip_diff = np.abs(angles['left_hip'] - angles['right_hip'])
    analysis['symmetry'] = {
        'knee_mean_diff': np.mean(knee_diff),
        'knee_max_diff': np.max(knee_diff),
        'hip_mean_diff': np.mean(hip_diff),
        'hip_max_diff': np.max(hip_diff),
    }
    
    # 手臂运动
    wrist_left = flight_kp[:, KP['left_wrist']]
    wrist_right = flight_kp[:, KP['right_wrist']]
    shoulder_y = (flight_kp[:, KP['left_shoulder'], 1] + flight_kp[:, KP['right_shoulder'], 1]) / 2
    
    analysis['arm'] = {
        'left_height': shoulder_y - wrist_left[:, 1],
        'right_height': shoulder_y - wrist_right[:, 1],
    }
    
    return analysis


# 可视化
def plot_preprocessing_comparison(frames, kp_raw, kp_clean, kp_smooth, athlete_id, save_dir):
    """绘制预处理前后对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选择脚踝Y作为示例
    raw_y = (kp_raw[:, KP['left_ankle'], 1] + kp_raw[:, KP['right_ankle'], 1]) / 2
    clean_y = (kp_clean[:, KP['left_ankle'], 1] + kp_clean[:, KP['right_ankle'], 1]) / 2
    smooth_y = (kp_smooth[:, KP['left_ankle'], 1] + kp_smooth[:, KP['right_ankle'], 1]) / 2
    
    # 左上：原始 vs 清洗
    ax = axes[0, 0]
    ax.plot(frames, raw_y, 'r-', alpha=0.5, linewidth=1, label='Raw')
    ax.plot(frames, clean_y, 'b-', linewidth=1.5, label='Cleaned')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ankle Y (pixels)')
    ax.set_title('Raw vs Cleaned Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右上：清洗 vs 平滑
    ax = axes[0, 1]
    ax.plot(frames, clean_y, 'b-', alpha=0.5, linewidth=1, label='Cleaned')
    ax.plot(frames, smooth_y, 'g-', linewidth=2, label='Smoothed')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ankle Y (pixels)')
    ax.set_title('Cleaned vs Smoothed Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 左下：速度对比
    ax = axes[1, 0]
    vel_clean = np.gradient(clean_y)
    vel_smooth = np.gradient(smooth_y)
    ax.plot(frames, vel_clean, 'b-', alpha=0.5, linewidth=1, label='Cleaned')
    ax.plot(frames, vel_smooth, 'g-', linewidth=2, label='Smoothed')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (pixels/frame)')
    ax.set_title('Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右下：异常值标记
    ax = axes[1, 1]
    outlier_mask = detect_outliers_iqr(raw_y, k=2.0) | (raw_y < 1)
    ax.plot(frames, raw_y, 'b-', linewidth=1, label='Raw')
    ax.scatter(frames[outlier_mask], raw_y[outlier_mask], c='red', s=50, 
               zorder=5, label=f'Outliers ({np.sum(outlier_mask)})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ankle Y (pixels)')
    ax.set_title('Outlier Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Athlete {athlete_id}: Preprocessing Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f'athlete{athlete_id}_preprocessing.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_detection_result(frames, keypoints, takeoff, landing, info, athlete_id, save_dir):
    """绘制检测结果"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 图1: 脚部Y坐标
    ax = axes[0]
    ax.plot(frames, info['foot_y_raw'], 'b-', alpha=0.3, linewidth=1, label='Raw')
    ax.plot(frames, info['foot_y_smooth'], 'b-', linewidth=2, label='Smoothed')
    ax.axhline(y=info['baseline_y'], color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax.axhline(y=info['baseline_y'] - info['height_threshold'], color='orange', 
               linestyle=':', alpha=0.7, label='Threshold')
    ax.axvline(x=takeoff, color='green', linestyle='--', linewidth=2, 
               label=f'Takeoff ({takeoff}±{info["uncertainty_frames"]})')
    ax.axvline(x=landing, color='red', linestyle='--', linewidth=2,
               label=f'Landing ({landing}±{info["uncertainty_frames"]})')
    ax.fill_between(frames, info['foot_y_smooth'], info['baseline_y'],
                    where=(frames >= takeoff) & (frames <= landing),
                    alpha=0.3, color='yellow')
    ax.scatter([info['flight_peak']], [info['foot_y_smooth'][info['flight_peak']]], 
               s=100, c='gold', marker='*', zorder=5, label='Peak')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Foot Y (pixels)')
    ax.set_title(f'Athlete {athlete_id}: Jump Phase Detection')
    ax.legend(loc='upper right', fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # 添加信息文本
    flight_time = (landing - takeoff + 1) / FPS
    height = info['baseline_y'] - np.min(info['foot_y_smooth'][takeoff:landing+1])
    ax.text(0.02, 0.98, 
            f'Flight: {flight_time:.3f}±{1/FPS:.3f}s\nHeight: {height:.1f}px',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 图2: 速度
    ax = axes[1]
    ax.plot(frames, info['velocity'], 'orange', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=info['vel_threshold'], color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=-info['vel_threshold'], color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=takeoff, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=landing, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (pixels/s)')
    ax.set_title('Vertical Velocity (negative = upward)')
    ax.grid(True, alpha=0.3)
    
    # 图3: 加速度
    ax = axes[2]
    ax.plot(frames, info['acceleration'], 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=takeoff, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=landing, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Acceleration (pixels/s²)')
    ax.set_title('Vertical Acceleration')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'athlete{athlete_id}_detection.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_flight_analysis(keypoints, takeoff, landing, analysis, athlete_id, save_dir):
    """绘制滞空分析"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    relative_frames = np.arange(analysis['n_frames'])
    peak_idx = analysis['peak']['relative_frame']
    
    # 图1: 质心轨迹（加权 vs 简化）
    ax = axes[0, 0]
    com_w = analysis['com']['weighted']
    com_s = analysis['com']['simple']
    ax.plot(com_w['x'], com_w['y'], 'b-', linewidth=2.5, label='Weighted CoM')
    ax.plot(com_s['x'], com_s['y'], 'g--', linewidth=1.5, alpha=0.7, label='Simple CoM')
    ax.scatter(com_w['x'][0], com_w['y'][0], s=150, c='green', marker='o', zorder=5, label='Takeoff')
    ax.scatter(com_w['x'][-1], com_w['y'][-1], s=150, c='red', marker='o', zorder=5, label='Landing')
    ax.scatter(com_w['x'][peak_idx], com_w['y'][peak_idx], s=200, c='gold', marker='*', zorder=5, label='Peak')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Center of Mass Trajectory')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # 图2: 膝关节角度
    ax = axes[0, 1]
    angles = analysis['angles']
    ax.plot(relative_frames, angles['left_knee'], 'b-', linewidth=2, label='Left')
    ax.plot(relative_frames, angles['right_knee'], 'r--', linewidth=2, label='Right')
    ax.fill_between(relative_frames, angles['left_knee'], angles['right_knee'], alpha=0.2)
    ax.axvline(x=peak_idx, color='gold', linestyle=':', linewidth=2)
    ax.set_xlabel('Frame (relative)')
    ax.set_ylabel('Angle (°)')
    ax.set_title(f'Knee Angles (L-R diff: {analysis["symmetry"]["knee_mean_diff"]:.1f}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图3: 髋关节角度
    ax = axes[0, 2]
    ax.plot(relative_frames, angles['left_hip'], 'b-', linewidth=2, label='Left')
    ax.plot(relative_frames, angles['right_hip'], 'r--', linewidth=2, label='Right')
    ax.fill_between(relative_frames, angles['left_hip'], angles['right_hip'], alpha=0.2)
    ax.axvline(x=peak_idx, color='gold', linestyle=':', linewidth=2)
    ax.set_xlabel('Frame (relative)')
    ax.set_ylabel('Angle (°)')
    ax.set_title(f'Hip Angles (L-R diff: {analysis["symmetry"]["hip_mean_diff"]:.1f}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图4: 踝关节角度
    ax = axes[1, 0]
    ax.plot(relative_frames, angles['left_ankle'], 'b-', linewidth=2, label='Left')
    ax.plot(relative_frames, angles['right_ankle'], 'r--', linewidth=2, label='Right')
    ax.axvline(x=peak_idx, color='gold', linestyle=':', linewidth=2)
    ax.set_xlabel('Frame (relative)')
    ax.set_ylabel('Angle (°)')
    ax.set_title('Ankle Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图5: 躯干角度
    ax = axes[1, 1]
    ax.plot(relative_frames, angles['trunk'], 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=peak_idx, color='gold', linestyle=':', linewidth=2)
    ax.set_xlabel('Frame (relative)')
    ax.set_ylabel('Angle (°)')
    ax.set_title('Trunk Inclination (+ = forward)')
    ax.grid(True, alpha=0.3)
    
    # 图6: 手臂位置
    ax = axes[1, 2]
    arm = analysis['arm']
    ax.plot(relative_frames, arm['left_height'], 'b-', linewidth=2, label='Left')
    ax.plot(relative_frames, arm['right_height'], 'r--', linewidth=2, label='Right')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, label='Shoulder')
    ax.axvline(x=peak_idx, color='gold', linestyle=':', linewidth=2)
    ax.set_xlabel('Frame (relative)')
    ax.set_ylabel('Height above shoulder (px)')
    ax.set_title('Arm Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Athlete {athlete_id}: Flight Phase Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f'athlete{athlete_id}_flight.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_skeleton_sequence(keypoints, takeoff, landing, athlete_id, save_dir, n_show=8):
    """绘制骨架序列（包括起跳前后）"""
    # 扩展范围：起跳前2帧，落地后2帧
    start = max(0, takeoff - 2)
    end = min(len(keypoints) - 1, landing + 2)
    
    indices = np.linspace(start, end, n_show, dtype=int)
    
    fig, axes = plt.subplots(1, n_show, figsize=(2.5*n_show, 6))
    
    bones = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 31),
        (24, 26), (26, 28), (28, 32),
    ]
    
    for ax, frame_idx in zip(axes, indices):
        kp = keypoints[frame_idx]
        
        # 根据阶段着色
        if frame_idx < takeoff:
            color = 'blue'
            phase = 'Pre'
        elif frame_idx > landing:
            color = 'red'
            phase = 'Post'
        else:
            color = 'green'
            phase = 'Flight'
        
        for start_idx, end_idx in bones:
            ax.plot([kp[start_idx, 0], kp[end_idx, 0]], 
                   [kp[start_idx, 1], kp[end_idx, 1]], 
                   f'{color[0]}-', linewidth=2)
        
        ax.scatter(kp[:, 0], kp[:, 1], c=color, s=15, zorder=5)
        ax.set_title(f'F{frame_idx} ({phase})', fontsize=10)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.suptitle(f'Athlete {athlete_id}: Skeleton Sequence', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f'athlete{athlete_id}_skeleton.png', dpi=150, bbox_inches='tight')
    plt.close()


# 报告生成
def generate_report(athlete_id, score, takeoff, landing, analysis, detection_info, fps=30):
    """生成详细报告（含不确定性）"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"运动者 {athlete_id} 滞空阶段分析报告")
    lines.append(f"跳远成绩: {score}")
    lines.append("=" * 60)
    
    # 时间信息
    unc = detection_info['uncertainty_frames']
    lines.append(f"\n【时间信息】")
    lines.append(f"  起跳帧: {takeoff} ± {unc} (t = {takeoff/fps:.3f}s)")
    lines.append(f"  落地帧: {landing} ± {unc} (t = {landing/fps:.3f}s)")
    lines.append(f"  滞空时间: {analysis['flight_time']:.3f} ± {analysis['flight_time_uncertainty']:.3f}s")
    lines.append(f"  最高点: 起跳后第 {analysis['peak']['relative_frame']} 帧")
    
    # 质心运动
    lines.append(f"\n【质心运动】(加权计算)")
    com = analysis['com']
    lines.append(f"  水平位移: {com['x_displacement']:.1f} pixels")
    lines.append(f"  垂直位移: {com['y_displacement']:.1f} pixels")
    
    if 'initial_velocity' in analysis:
        iv = analysis['initial_velocity']
        lines.append(f"  初始速度: {iv['magnitude']:.1f} pixels/s")
        lines.append(f"  起跳角度: {iv['angle']:.1f}°")
    
    # 关节角度
    lines.append(f"\n【关节角度变化】")
    angles = analysis['angles']
    
    for joint in ['knee', 'hip', 'ankle']:
        left = angles[f'left_{joint}']
        right = angles[f'right_{joint}']
        avg_min = (np.min(left) + np.min(right)) / 2
        avg_max = (np.max(left) + np.max(right)) / 2
        lines.append(f"  {joint.capitalize()}: {avg_min:.1f}° ~ {avg_max:.1f}° (范围: {avg_max-avg_min:.1f}°)")
    
    trunk = angles['trunk']
    lines.append(f"  躯干倾角: {np.min(trunk):.1f}° ~ {np.max(trunk):.1f}°")
    
    # 对称性
    lines.append(f"\n【左右对称性】")
    sym = analysis['symmetry']
    lines.append(f"  膝关节差异: 平均{sym['knee_mean_diff']:.1f}°, 最大{sym['knee_max_diff']:.1f}°")
    lines.append(f"  髋关节差异: 平均{sym['hip_mean_diff']:.1f}°, 最大{sym['hip_max_diff']:.1f}°")
    
    # 手臂
    lines.append(f"\n【手臂摆动】")
    arm = analysis['arm']
    left_range = np.max(arm['left_height']) - np.min(arm['left_height'])
    right_range = np.max(arm['right_height']) - np.min(arm['right_height'])
    lines.append(f"  左臂幅度: {left_range:.1f} pixels")
    lines.append(f"  右臂幅度: {right_range:.1f} pixels")
    
    # 检测参数
    lines.append(f"\n【检测参数】")
    lines.append(f"  地面估计: {detection_info['ground_y']:.1f} pixels")
    lines.append(f"  站立基准: {detection_info['baseline_y']:.1f} ± {detection_info['baseline_std']:.1f} pixels")
    lines.append(f"  高度阈值: {detection_info['height_threshold']:.1f} pixels")
    lines.append(f"  速度阈值: {detection_info['vel_threshold']:.1f} pixels/s")
    
    return '\n'.join(lines)


def export_results_json(athlete_id, results, save_dir):
    """导出结果为JSON（便于后续问题复用）"""
    export_data = {
        'athlete_id': results['athlete_id'],
        'score': results['score'],
        'takeoff': int(results['takeoff']),
        'landing': int(results['landing']),
        'valid_end': int(results['valid_end']),
        'flight_time': results['analysis']['flight_time'],
        'flight_time_uncertainty': results['analysis']['flight_time_uncertainty'],
        'com_x_displacement': float(results['analysis']['com']['x_displacement']),
        'com_y_displacement': float(results['analysis']['com']['y_displacement']),
        'peak_frame': int(results['analysis']['peak']['frame']),
    }
    
    if 'initial_velocity' in results['analysis']:
        export_data['initial_velocity'] = {
            k: float(v) for k, v in results['analysis']['initial_velocity'].items()
        }
    
    # 角度统计
    export_data['angles'] = {}
    for joint in ['knee', 'hip', 'ankle', 'trunk']:
        if joint == 'trunk':
            data = results['analysis']['angles']['trunk']
            export_data['angles'][joint] = {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
            }
        else:
            left = results['analysis']['angles'][f'left_{joint}']
            right = results['analysis']['angles'][f'right_{joint}']
            export_data['angles'][joint] = {
                'left_min': float(np.min(left)),
                'left_max': float(np.max(left)),
                'right_min': float(np.min(right)),
                'right_max': float(np.max(right)),
                'mean_diff': float(np.mean(np.abs(left - right))),
            }
    
    with open(save_dir / f'athlete{athlete_id}_results.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return export_data


# 主函数
def analyze_athlete(filepath, athlete_id, score, save_dir, fps=30):
    """分析单个运动者"""
    print(f"\n{'='*60}")
    print(f"分析运动者 {athlete_id} (成绩: {score})")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print("\n[1. 加载数据]")
    frames_raw, kp_raw = load_data(filepath)
    print_data_statistics(kp_raw, "原始数据")
    
    # 2. 预处理
    print("\n[2. 预处理]")
    kp_smooth, kp_clean = preprocess_keypoints(kp_raw, verbose=True)
    print_data_statistics(kp_smooth, "预处理后")
    
    # 3. 有效范围检测
    print("\n[3. 有效范围检测]")
    valid_end, range_info = find_valid_range(kp_smooth, verbose=True)
    
    # 裁剪数据
    frames = frames_raw[:valid_end+1]
    keypoints = kp_smooth[:valid_end+1]
    kp_clean = kp_clean[:valid_end+1]
    kp_raw_trimmed = kp_raw[:valid_end+1]
    
    # 4. 起跳/落地检测
    print("\n[4. 起跳/落地检测]")
    takeoff, landing, detection_info = detect_jump_phases_stateful(keypoints, fps=fps, verbose=True)
    print(f"  起跳帧: {takeoff} ± {detection_info['uncertainty_frames']}")
    print(f"  落地帧: {landing} ± {detection_info['uncertainty_frames']}")
    print(f"  滞空时间: {(landing-takeoff+1)/fps:.3f}s")
    
    # 5. 滞空分析
    print("\n[5. 滞空分析]")
    analysis = analyze_flight_phase(keypoints, takeoff, landing, fps=fps)
    print(f"  质心水平位移: {analysis['com']['x_displacement']:.1f} px")
    print(f"  左右膝关节差异: {analysis['symmetry']['knee_mean_diff']:.1f}°")
    
    # 6. 可视化
    print("\n[6. 生成图表]")
    plot_preprocessing_comparison(frames, kp_raw_trimmed, kp_clean, keypoints, athlete_id, save_dir)
    print(f"  Saved: athlete{athlete_id}_preprocessing.png")
    
    plot_detection_result(frames, keypoints, takeoff, landing, detection_info, athlete_id, save_dir)
    print(f"  Saved: athlete{athlete_id}_detection.png")
    
    plot_flight_analysis(keypoints, takeoff, landing, analysis, athlete_id, save_dir)
    print(f"  Saved: athlete{athlete_id}_flight.png")
    
    plot_skeleton_sequence(keypoints, takeoff, landing, athlete_id, save_dir)
    print(f"  Saved: athlete{athlete_id}_skeleton.png")
    
    # 7. 报告
    print("\n[7. 生成报告]")
    report = generate_report(athlete_id, score, takeoff, landing, analysis, detection_info, fps=fps)
    print(report)
    
    with open(save_dir / f'athlete{athlete_id}_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 8. 导出JSON
    results = {
        'athlete_id': athlete_id,
        'score': score,
        'takeoff': takeoff,
        'landing': landing,
        'valid_end': valid_end,
        'analysis': analysis,
        'keypoints': keypoints,
    }
    export_results_json(athlete_id, results, save_dir)
    print(f"  Saved: athlete{athlete_id}_results.json")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("问题1: 起跳/落地检测 + 滞空阶段分析")
    print("=" * 60)
    
    athletes = [
        {"id": 1, "file": INPUT_DIR + "Athlete_01_PositionData.xlsx", "score": "1.58米"},
        {"id": 2, "file": INPUT_DIR + "Athlete_02_PositionData.xlsx", "score": "1.15米"},
    ]
    
    results = {}
    
    for athlete in athletes:
        filepath = Path(athlete["file"])
        if filepath.exists():
            results[athlete["id"]] = analyze_athlete(
                filepath, athlete["id"], athlete["score"], OUTPUT_DIR, fps=FPS
            )
        else:
            print(f"Warning: {filepath} not found")
    
    # 对比
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("两位运动者对比")
        print("=" * 60)
        
        comparison = []
        for aid in [1, 2]:
            r = results[aid]
            a = r['analysis']
            row = {
                'ID': aid,
                '成绩': r['score'],
                '滞空(s)': f"{a['flight_time']:.3f}",
                '水平位移(px)': f"{a['com']['x_displacement']:.0f}",
            }
            if 'initial_velocity' in a:
                row['起跳速度(px/s)'] = f"{a['initial_velocity']['magnitude']:.0f}"
                row['起跳角度(°)'] = f"{a['initial_velocity']['angle']:.1f}"
            row['膝对称性(°)'] = f"{a['symmetry']['knee_mean_diff']:.1f}"
            comparison.append(row)
        
        # 打印对比表
        print("\n" + "-" * 60)
        headers = list(comparison[0].keys())
        print(" | ".join(f"{h:>12}" for h in headers))
        print("-" * 60)
        for row in comparison:
            print(" | ".join(f"{str(row[h]):>12}" for h in headers))
    
    print(f"\n所有结果已保存至: {OUTPUT_DIR.absolute()}")
    return results


if __name__ == "__main__":
    main()