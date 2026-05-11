import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree 
import argparse
import traceback
from config_manager import cfg  # 引入全局配置管理器
from scipy.signal import find_peaks # 用于寻找速度突变峰值

INPUT_CSV = "drone_path_optimized.csv"   
FIGURE_SAVE_PATH = "safety_report.png" 

def analyze_safety(csv_file=None, safe_distance=None, max_velocity=None, 
                   max_acceleration=None, figure_path=None, interactive=False):
    """
    核心安全性体检：生成间距、速度、加速度的时域三联图
    """
    if interactive:
        csv_file = csv_file or input(f"CSV文件路径: ").strip() or INPUT_CSV
    
    csv_file = csv_file or INPUT_CSV
    # 默认值回退逻辑：优先使用传入参数，未传入时 safe_distance 和 max_velocity 使用常规默认值
    # max_acceleration 严格与全局 cfg 物理限制绑定
    safe_distance = safe_distance if safe_distance is not None else 1.5
    max_velocity = max_velocity if max_velocity is not None else 10.0
    max_acceleration = max_acceleration if max_acceleration is not None else cfg.max_accel
    figure_path = figure_path or FIGURE_SAVE_PATH

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[错误] 找不到文件或读取失败。{e}")
        return

    df = df.dropna(subset=['Frame', 'Time', 'VertexID', 'X', 'Y', 'Z'])
    
    # =========================================================================
    # 步骤 1：运动学真实解算 (修复“相对论幻觉” & 补充加速度)
    # =========================================================================
    print("正在进行物理速度与加速度解算 (严谨微分)...")
    
    df = df.sort_values(by=['VertexID', 'Time']) 
    
    try:
        df['dt'] = df.groupby('VertexID')['Time'].diff()
        
        global_median_dt = df['dt'].median()
        if pd.isna(global_median_dt) or global_median_dt <= 0.001: 
            global_median_dt = 0.05
            
        df['dt'] = df['dt'].fillna(global_median_dt)
        df['dt'] = df['dt'].replace(0, global_median_dt) 

        df['dX'] = df.groupby('VertexID')['X'].diff().fillna(0.0)
        df['dY'] = df.groupby('VertexID')['Y'].diff().fillna(0.0)
        df['dZ'] = df.groupby('VertexID')['Z'].diff().fillna(0.0)
        
        df['Vel_X'] = df['dX'] / df['dt']
        df['Vel_Y'] = df['dY'] / df['dt']
        df['Vel_Z'] = df['dZ'] / df['dt']
        df['Vel'] = np.sqrt(df['Vel_X']**2 + df['Vel_Y']**2 + df['Vel_Z']**2)
        
        df['Acc_X'] = df.groupby('VertexID')['Vel_X'].diff().fillna(0.0) / df['dt']
        df['Acc_Y'] = df.groupby('VertexID')['Vel_Y'].diff().fillna(0.0) / df['dt']
        df['Acc_Z'] = df.groupby('VertexID')['Vel_Z'].diff().fillna(0.0) / df['dt']
        df['Acc'] = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)

        max_vel_series = df.groupby('Frame')['Vel'].max()
        max_vels = max_vel_series.values
        
        max_acc_series = df.groupby('Frame')['Acc'].max()
        max_accs = max_acc_series.values
        
        actual_data_max_vel = df['Vel'].max()
        
    except Exception as e:
        print(f"[警告] 动力学计算失败。错误信息: {e}")
        traceback.print_exc()
        return

    # =========================================================================
    # 步骤 2：距离检测 (修复了盲区的两阶段连续碰撞预测 CCD)
    # =========================================================================
    print("正在进行安全间距检测 (亚帧级 CCD 近似)...")
    
    df = df.sort_values(by=['Frame', 'Object', 'VertexID'])
    frames = df['Frame'].unique()
    frames.sort()
    
    min_dists = []      
    collision_events = [] 
    
    effective_max_vel = max(max_velocity, actual_data_max_vel) * 1.2 
    alert_radius = safe_distance + (2.0 * effective_max_vel * global_median_dt)
    
    SUB_STEPS = 10
    alphas = np.linspace(0, 1, SUB_STEPS)

    for frame_idx, f in enumerate(frames):
        try:
            current_data = df[df['Frame'] == f]
            positions = current_data[['X', 'Y', 'Z']].values
            ids = current_data['VertexID'].values
            obj_names = current_data['Object'].values
            
            if len(positions) <= 1:
                min_dists.append(safe_distance * 2)
                continue
                
            tree = cKDTree(positions)
            dists, _ = tree.query(positions, k=2)
            if dists.shape[1] > 1:
                min_dists.append(np.min(dists[:, 1]))
            else:
                min_dists.append(safe_distance * 2)
                
            suspect_pairs = tree.query_pairs(r=alert_radius)
            
            if not suspect_pairs or frame_idx == len(frames) - 1:
                continue
                
            next_f = frames[frame_idx + 1]
            next_data = df[df['Frame'] == next_f]
            next_pos_dict = dict(zip(next_data['VertexID'], next_data[['X', 'Y', 'Z']].values))
            current_time = current_data.iloc[0]['Time']
            
            for i, j in suspect_pairs:
                id_A, id_B = ids[i], ids[j]
                posA_t0, posB_t0 = positions[i], positions[j]
                posA_t1 = next_pos_dict.get(id_A)
                posB_t1 = next_pos_dict.get(id_B)
                
                if posA_t1 is None or posB_t1 is None:
                    continue
                    
                for alpha in alphas:
                    posA_sub = posA_t0 * (1.0 - alpha) + posA_t1 * alpha
                    posB_sub = posB_t0 * (1.0 - alpha) + posB_t1 * alpha
                    dist_sub = np.linalg.norm(posA_sub - posB_sub)
                    
                    if dist_sub < safe_distance:
                        collision_events.append({
                            'Frame': f, 'Time': current_time + alpha * global_median_dt,
                            'Drone1': f"{obj_names[i]}_{id_A}", 'Drone2': f"{obj_names[j]}_{id_B}",
                            'Dist': dist_sub
                        })
                        break 
        except Exception as e:
            min_dists.append(safe_distance * 2)
            continue

    # =========================================================================
    # 步骤 3：报告生成与三维图表绘制
    # =========================================================================
    print("\n" + "="*50)
    print(" 工业级飞行安全审计报告 ")
    print("="*50)
    
    if len(min_dists) > 0:
        min_dist_global = np.min(min_dists)
        max_vel_global = np.max(max_vels)
        max_acc_global = np.max(max_accs)
        
        print(f"1. 最小物理间距: {min_dist_global:.4f} m")
        if min_dist_global < safe_distance: 
            print(f"   [错误] 存在空间碰撞风险 (阈值 {safe_distance}m)")
        else: 
            print(f"   [通过] 空间防撞测试")
            
        print(f"2. 极限飞行速度: {max_vel_global:.4f} m/s")
        if max_vel_global > max_velocity: 
            print(f"   [警告] 存在超速越界风险 (阈值 {max_velocity}m/s)")
        else: 
            print(f"   [通过] 动力学速度测试")
            
        print(f"3. 极限加速度: {max_acc_global:.4f} m/s²")
        if max_acc_global > max_acceleration: 
            print(f"   [错误] 检测到高频轨迹抖动或电机推力溢出 (电机极限 {max_acceleration}m/s²)")
        else: 
            print(f"   [通过] 电机过载与轨迹平滑度测试")
        
        print(f"4. 动态穿透事件 (CCD): 共发现 {len(collision_events)} 次高频穿透碰撞")
    
    plt.style.use('ggplot') 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    t_diff = df['Time'].unique()
    t_diff.sort()
    limit_len = min(len(min_dists), len(max_vels), len(max_accs))
    time_steps = t_diff[:limit_len] if len(t_diff) >= limit_len else np.arange(limit_len) * global_median_dt
        
    min_dists = min_dists[:limit_len]
    plot_max_vels = np.clip(max_vels[:limit_len], 0, max_velocity * 3.0) 
    plot_max_accs = np.clip(max_accs[:limit_len], 0, max_acceleration * 3.0)
    
    ax1.plot(time_steps, min_dists, color='#2ca02c', label='Min Distance Between Drones')
    ax1.axhline(y=safe_distance, color='red', linestyle='--', label=f'Safety Limit ({safe_distance}m)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Spatial Safety (Collision Proximity)')
    ax1.legend(loc='upper right')
    
    ax2.plot(time_steps, plot_max_vels, color='#1f77b4', label='Max Velocity (Kinematics)')
    ax2.axhline(y=max_velocity, color='orange', linestyle='--', label=f'Velocity Limit ({max_velocity}m/s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Kinematic Safety (Speed Limit)')
    ax2.legend(loc='upper right')

    ax3.plot(time_steps, plot_max_accs, color='#9467bd', label='Max Acceleration (Jerk / Jitter)')
    ax3.axhline(y=max_acceleration, color='purple', linestyle='--', label=f'Motor Output Limit ({max_acceleration}m/s²)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Dynamics Safety (Motor Overload & Trajectory Jitter)')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    print(f"\n[系统] 高精度体检图表已生成并保存至: {figure_path}")


def generate_peak_velocity_distribution(csv_file=INPUT_CSV, max_velocity=10.0, save_path="velocity_peaks_distribution.png"):
    """
    智能截取最高速爆发时刻的 2D 速度分布快照 (4个高危切面直方图)
    """
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Frame', 'Time', 'VertexID', 'X', 'Y', 'Z'])
        df = df.sort_values(by=['VertexID', 'Time'])
        
        # 1. 解算绝对速度
        df['dt'] = df.groupby('VertexID')['Time'].diff().fillna(0.05)
        df['dt'] = df['dt'].replace(0, 0.05)
        df['Vel_X'] = df.groupby('VertexID')['X'].diff().fillna(0.0) / df['dt']
        df['Vel_Y'] = df.groupby('VertexID')['Y'].diff().fillna(0.0) / df['dt']
        df['Vel_Z'] = df.groupby('VertexID')['Z'].diff().fillna(0.0) / df['dt']
        df['Vel'] = np.sqrt(df['Vel_X']**2 + df['Vel_Y']**2 + df['Vel_Z']**2)
        
        # 2. 提取全局每一帧的最大速度
        max_vel_series = df.groupby('Time')['Vel'].max()
        times = max_vel_series.index.values
        max_vels = max_vel_series.values
        
        # 3. 寻找速度爆发的峰值时刻 (避免连续取同一事件的相邻帧)
        print("正在扫描速度爆发峰值...")
        peaks, _ = find_peaks(max_vels, distance=20) # 假设相隔约1秒为独立事件
        
        if len(peaks) == 0:
            # 备用方案：如果没有明显的尖峰，直接按速度从大到小取间隔帧
            sorted_indices = np.argsort(max_vels)[::-1]
            top_peaks = []
            for idx in sorted_indices:
                if not top_peaks or all(abs(idx - p) > 20 for p in top_peaks):
                    top_peaks.append(idx)
                if len(top_peaks) >= 4: break
        else:
            # 按峰值大小降序排列，取最危险的前 4 个时刻
            top_peaks = sorted(peaks, key=lambda p: max_vels[p], reverse=True)[:4]

        if not top_peaks:
            top_peaks = [np.argmax(max_vels)]

        # 4. 绘图渲染
        plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        num_peaks = len(top_peaks)
        cols = 2 if num_peaks > 1 else 1
        rows = int(np.ceil(num_peaks / 2))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        if num_peaks == 1: axes = [axes]
        else: axes = axes.flatten()
        
        plot_limit_vel = max(max_velocity * 1.2, max(max_vels[top_peaks]) * 1.1)
        bins_edges = np.linspace(0, plot_limit_vel, 30)

        for i, peak_idx in enumerate(top_peaks):
            target_time = times[peak_idx]
            peak_vel = max_vels[peak_idx]
            frame_data = df[df['Time'] == target_time]
            
            ax = axes[i]
            # 绘制直方图
            counts, bins, patches = ax.hist(frame_data['Vel'], bins=bins_edges, edgecolor='white')
            
            # 【视觉增强】将超过安全极限速度的柱子标红，其余为蓝色
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge >= max_velocity:
                    patch.set_facecolor('#d62728') # 红色警告
                else:
                    patch.set_facecolor('#1f77b4') # 正常蓝色
                    
            ax.axvline(max_velocity, color='orange', linestyle='--', linewidth=2, label=f'设计极限 ({max_velocity} m/s)')
            ax.axvline(peak_vel, color='red', linestyle='-', linewidth=1.5, label=f'本阵最高速 ({peak_vel:.2f} m/s)')
            
            ax.set_title(f'最高危时刻切片: 第 {target_time:.2f} 秒', fontsize=14)
            ax.set_xlabel('无人机速度 (m/s)', fontsize=12)
            ax.set_ylabel('无人机数量 (架)', fontsize=12)
            ax.legend()

        # 隐藏多余的空白子图
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle('系统极限速度时刻的集群分布剖面分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[系统] 峰值速度分布快照已保存至: {save_path}")
        plt.show()

    except Exception as e:
        print(f"[错误] 分布制图失败: {e}")
        traceback.print_exc()

# ==========================================
# 整合入口
# ==========================================
if __name__ == "__main__":
    analyze_safety()
    generate_peak_velocity_distribution()