
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree 
import sys
import argparse
import traceback

INPUT_CSV = "drone_path_optimized.csv"   
SAFE_DISTANCE = 1.5            
MAX_VELOCITY = 10.0            
FIGURE_SAVE_PATH = "safety_report.png" 

def analyze_safety(csv_file=None, safe_distance=None, max_velocity=None, 
                   figure_path=None, interactive=False):
    if interactive:
        csv_file = csv_file or input(f"CSV文件路径: ").strip() or INPUT_CSV
    
    csv_file = csv_file or INPUT_CSV
    safe_distance = safe_distance if safe_distance is not None else SAFE_DISTANCE
    max_velocity = max_velocity if max_velocity is not None else MAX_VELOCITY
    figure_path = figure_path or FIGURE_SAVE_PATH

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"错误：找不到文件或读取失败。{e}")
        return

    # 数据强制排序与清洗，防止各种因为 NaN 引起的血案
    df = df.dropna(subset=['Frame', 'Time', 'VertexID', 'X', 'Y', 'Z'])
    df = df.sort_values(by=['Frame', 'Object', 'VertexID'])
    frames = df['Frame'].unique()
    
    # 提取时间轴计算步长，并防备异常脏数据导致的极小值
    t_diff = df['Time'].unique()
    t_diff.sort()
    dt = np.median(np.diff(t_diff))
    if np.isnan(dt) or dt <= 0.001: dt = 0.05 

    min_dists =[]      
    max_vels = []       
    collision_events =[] 
    
    # ================= 1. 距离检测 (使用极速 KDTree) =================
    print("正在进行安全间距检测...")
    for f in frames:
        current_data = df[df['Frame'] == f]
        positions = current_data[['X', 'Y', 'Z']].values
        ids = current_data['VertexID'].values
        obj_names = current_data['Object'].values
        
        if len(positions) > 1:
            tree = cKDTree(positions)
            dists, idxs = tree.query(positions, k=2)
            
            if dists.shape[1] > 1:
                min_d = np.min(dists[:, 1])
                min_dists.append(min_d)
                
                if min_d < safe_distance:
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

    print("正在进行物理速度检测...")
    df = df.sort_values(by=['VertexID', 'Time']) 
    
    try:
        df['dX'] = df.groupby('VertexID')['X'].diff().fillna(0.0)
        df['dY'] = df.groupby('VertexID')['Y'].diff().fillna(0.0)
        df['dZ'] = df.groupby('VertexID')['Z'].diff().fillna(0.0)
        
        df['Vel'] = np.sqrt(df['dX']**2 + df['dY']**2 + df['dZ']**2) / dt
        

        max_vel_series = df.groupby('Frame')['Vel'].max()
        max_vels = max_vel_series.values
        
    except Exception as e:
        print(f"警告：速度计算失败，已启用安全保底。错误信息: {e}")
        traceback.print_exc()
        max_vels =[0.0] * len(min_dists)

    plot_max_vels = np.clip(max_vels, 0, max_velocity * 5.0) 

    print("\n" + "="*50)
    print(" 安全性验证 ")
    print("="*50)
    if len(min_dists) > 0:
        min_dist_global = np.min(min_dists)
        max_vel_global = np.max(max_vels)
        
        print(f"1. 最小间距: {min_dist_global:.4f} m")
        if min_dist_global < safe_distance: 
            print(f"  存在碰撞风险 (阈值 {safe_distance}m)")
        else: 
            print(f"  空间安全")
            
        print(f"2. 最大速度: {max_vel_global:.4f} m/s")
        if max_vel_global > max_velocity: 
            print(f"   存在超速风险 (阈值 {max_velocity}m/s)")
        else: 
            print(f"   动力学安全")
        
        print(f"3. 碰撞事件: 共发现 {len(collision_events)} 次瞬间越界")
    
    plt.style.use('ggplot') 
    plt.rcParams['axes.unicode_minus'] = False 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 强制数据长度对齐，避免因数组长度不同引发画图失败
    limit_len = min(len(min_dists), len(plot_max_vels))
    if len(t_diff) > limit_len:
        time_steps = t_diff[:limit_len]
    else:
        # 如果获取到的独特时间少于计算出的帧数，按索引生成虚拟时间轴
        time_steps = np.arange(limit_len) * dt
        
    min_dists = min_dists[:limit_len]
    plot_max_vels = plot_max_vels[:limit_len]
    
    ax1.plot(time_steps, min_dists, color='#2ca02c', label='Min Distance (KDTree)')
    ax1.axhline(y=safe_distance, color='red', linestyle='--', label=f'Safety Limit ({safe_distance}m)')
    ax1.set_ylabel('Distance (m)')
    ax1.legend(loc='upper right')
    
    ax2.plot(time_steps, plot_max_vels, color='#1f77b4', label='Max Velocity (Backward Diff)')
    ax2.axhline(y=max_velocity, color='orange', linestyle='--', label=f'Velocity Limit ({max_velocity}m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    print(f"图表已保存至: {figure_path}")

if __name__ == "__main__":
    analyze_safety()