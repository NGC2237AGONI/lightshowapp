
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree # V1.9: 极速 KDTree
from scipy.signal import savgol_filter # V1.9: 平滑导数
import sys
import argparse

INPUT_CSV = "drone_path_optimized.csv"   
SAFE_DISTANCE = 1.5            
MAX_VELOCITY = 10.0            
FIGURE_SAVE_PATH = "safety_report.png" 

def analyze_safety(csv_file=None, safe_distance=None, max_velocity=None, 
                   figure_path=None, interactive=False):
    # ... (参数处理保持不变) ...
    if interactive:
        csv_file = csv_file or input(f"CSV文件路径: ").strip() or INPUT_CSV
    
    csv_file = csv_file or INPUT_CSV
    safe_distance = safe_distance if safe_distance is not None else SAFE_DISTANCE
    max_velocity = max_velocity if max_velocity is not None else MAX_VELOCITY
    figure_path = figure_path or FIGURE_SAVE_PATH

    try:
        df = pd.read_csv(csv_file)
    except:
        print("错误：找不到文件。")
        return

    # 排序
    df = df.sort_values(by=['Frame', 'Object', 'VertexID'])
    frames = df['Frame'].unique()
    dt = df['Time'].diff().mean()
    if np.isnan(dt) or dt <= 0: dt = 0.05 

    min_dists = []      
    max_vels = []       
    collision_events = [] 
    
    # 1. 距离检测 (使用 KDTree)
    print("正在进行距离检测 (KDTree加速)...")
    for f in frames:
        current_data = df[df['Frame'] == f]
        positions = current_data[['X', 'Y', 'Z']].values
        ids = current_data['VertexID'].values
        obj_names = current_data['Object'].values
        
        if len(positions) > 1:
            tree = cKDTree(positions)
            # query 找最近邻 (k=2, 因为第一个是自己)
            dists, idxs = tree.query(positions, k=2)
            
            # dists[:, 1] 是到最近邻居的距离
            # 注意：如果只有1个点，k=2会报错，这里做了len判断
            if dists.shape[1] > 1:
                min_d = np.min(dists[:, 1])
                min_dists.append(min_d)
                
                if min_d < safe_distance:
                    # 获取详细碰撞对
                    pairs = tree.query_pairs(r=safe_distance)
                    for i, j in pairs:
                        dist_val = np.linalg.norm(positions[i] - positions[j])
                        collision_events.append({
                            'Frame': f,
                            'Time': current_data.iloc[0]['Time'],
                            'Drone1': f"{obj_names[i]}_{ids[i]}",
                            'Drone2': f"{obj_names[j]}_{ids[j]}",
                            'Dist': dist_val
                        })
            else:
                 min_dists.append(safe_distance * 2)
        else:
            min_dists.append(safe_distance * 2)

    # 2. 速度检测 (使用 SavGol)
    print("正在进行速度检测 (SavGol平滑)...")
    # 重新按 ID 排序以计算轨迹
    df_traj = df.sort_values(by=['VertexID', 'Frame'])
    
    # 定义求导函数
    def calc_velocity(group):
        if len(group) > 7:
            # 窗口7, 2阶多项式, 1阶导数, delta=dt
            vx = savgol_filter(group['X'], 7, 2, deriv=1, delta=dt)
            vy = savgol_filter(group['Y'], 7, 2, deriv=1, delta=dt)
            vz = savgol_filter(group['Z'], 7, 2, deriv=1, delta=dt)
            return np.sqrt(vx**2 + vy**2 + vz**2)
        else:
            return np.zeros(len(group)) # 样本太少无法计算

    # 对每个无人机计算速度序列
    # 这里的速度是针对每个点的每一帧的
    df_traj['Vel'] = df_traj.groupby('VertexID').apply(calc_velocity).reset_index(level=0, drop=True)
    
    # 聚合每一帧的最大速度
    max_vels = df_traj.groupby('Frame')['Vel'].max().values

    # 报告生成 (保持不变)
    print("\n" + "="*50)
    print("       🛡️ V1.9 安全性验证报告 🛡️")
    print("="*50)
    if len(min_dists) > 0:
        min_dist_global = np.min(min_dists)
        max_vel_global = np.max(max_vels)
        
        print(f"1. 最小间距: {min_dist_global:.4f} m")
        if min_dist_global < safe_distance: 
            print(f"  ⚠️ 存在碰撞风险 (阈值 {safe_distance}m)")
        else: 
            print(f"  ✅ 空间安全")
            
        print(f"2. 最大速度: {max_vel_global:.4f} m/s")
        if max_vel_global > max_velocity: 
            print(f"  ⚠️ 存在超速风险 (阈值 {max_velocity}m/s)")
        else: 
            print(f"  ✅ 动力学安全")
        
        print(f"3. 碰撞事件: 共 {len(collision_events)} 帧次")
    
    # 绘图 (保持不变)
    plt.style.use('ggplot') 
    plt.rcParams['axes.unicode_minus'] = False 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 注意：帧数对其
    time_steps = df['Time'].unique()[:len(min_dists)]
    
    ax1.plot(time_steps, min_dists, color='#2ca02c', label='Min Distance (KDTree)')
    ax1.axhline(y=safe_distance, color='red', linestyle='--', label=f'Limit ({safe_distance}m)')
    ax1.legend()
    
    ax2.plot(time_steps, max_vels, color='#1f77b4', label='Max Velocity (SavGol)')
    ax2.axhline(y=max_velocity, color='orange', linestyle='--', label=f'Limit ({max_velocity}m/s)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    print(f"图表已保存: {figure_path}")

if __name__ == "__main__":
    analyze_safety()