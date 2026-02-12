import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import sys
import argparse

# ================= 配置区域 =================
INPUT_CSV = "drone_path.csv"   # 默认轨迹文件
SAFE_DISTANCE = 1.5            # 默认安全距离阈值 (米)
MAX_VELOCITY = 10.0            # 默认最大速度限制 (米/秒)
FIGURE_SAVE_PATH = "safety_report.png" # 图表保存路径
# ===========================================

def analyze_safety(csv_file=None, safe_distance=None, max_velocity=None, 
                   figure_path=None, interactive=False):
    """
    安全分析函数（参数化版本）
    
    参数:
        csv_file: CSV文件路径（None则使用默认值）
        safe_distance: 安全距离阈值（None则使用默认值）
        max_velocity: 最大速度阈值（None则使用默认值）
        figure_path: 图表保存路径（None则使用默认值）
        interactive: 是否交互式输入参数
    """
    # 交互式输入（如果启用）
    if interactive:
        print("="*50)
        print("无人机表演安全分析 - 参数配置")
        print("="*50)
        
        csv_file = csv_file or input(f"CSV文件路径 (默认: {INPUT_CSV}): ").strip() or INPUT_CSV
        safe_dist_str = input(f"最小安全距离(米) (默认: {SAFE_DISTANCE}): ").strip()
        safe_distance = float(safe_dist_str) if safe_dist_str else SAFE_DISTANCE
        
        max_vel_str = input(f"最大允许速度(米/秒) (默认: {MAX_VELOCITY}): ").strip()
        max_velocity = float(max_vel_str) if max_vel_str else MAX_VELOCITY
        
        figure_path = figure_path or input(f"图表保存路径 (默认: {FIGURE_SAVE_PATH}): ").strip() or FIGURE_SAVE_PATH
    
    # 使用传入参数或默认值
    csv_file = csv_file or INPUT_CSV
    safe_distance = safe_distance if safe_distance is not None else SAFE_DISTANCE
    max_velocity = max_velocity if max_velocity is not None else MAX_VELOCITY
    figure_path = figure_path or FIGURE_SAVE_PATH
    print(f"正在加载数据: {csv_file} ...")
    try:
        # 使用 pandas 快速读取
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("错误：找不到文件。请先运行 drone_app.py 导出数据。")
        return
    except ImportError:
        print("错误：缺少 pandas 库。请运行: pip install pandas scipy matplotlib")
        return

    # 【学术严谨性修正】
    # 强制按帧、物体名、顶点ID排序，确保计算速度时前后帧的点是一一对应的
    print("整理数据顺序...")
    df = df.sort_values(by=['Frame', 'Object', 'VertexID'])

    # 获取所有帧号
    frames = df['Frame'].unique()
    total_frames = len(frames)
    
    # 自动计算时间步长 dt (取平均值以消除浮点误差)
    dt = df['Time'].diff().mean()
    # 如果只有一帧或读取失败，给个默认值防止除以0
    if np.isnan(dt) or dt <= 0: dt = 0.05 

    print(f"数据加载完毕，共 {total_frames} 帧，时间步长约 {dt:.4f}s")
    print("开始进行全流程安全检验...")

    # 存储统计数据
    min_dists = []      # 每一帧的全局最小距离
    max_vels = []       # 每一帧的全局最大速度
    collision_events = [] # 碰撞事件记录
    
    # 上一帧的位置 (用于算速度)
    prev_positions = None

    for f in frames:
        # 提取当前帧数据
        current_data = df[df['Frame'] == f]
        
        # 提取坐标矩阵 (N, 3)
        positions = current_data[['X', 'Y', 'Z']].values
        # 提取对应的ID，用于报错时告诉是哪个点
        ids = current_data['VertexID'].values
        obj_names = current_data['Object'].values
        
        #  1. 距离检测 (空间安全性)
        if len(positions) > 1:
            # 计算所有点对距离
            dists = pdist(positions)
            min_d = np.min(dists)
            min_dists.append(min_d)
            
            # 如果发现危险，记录详细信息
            if min_d < safe_distance:
                dist_matrix = squareform(dists)
                np.fill_diagonal(dist_matrix, np.inf) # 排除自己和自己
                
                # 找到所有小于阈值的索引对
                collision_indices = np.argwhere(dist_matrix < safe_distance)
                # 去重 (只保留 i < j 的情况)
                collision_indices = collision_indices[collision_indices[:, 0] < collision_indices[:, 1]]
                
                for idx1, idx2 in collision_indices:
                    collision_events.append({
                        'Frame': f,
                        'Time': current_data.iloc[0]['Time'],
                        'Drone1': f"{obj_names[idx1]}_{ids[idx1]}",
                        'Drone2': f"{obj_names[idx2]}_{ids[idx2]}",
                        'Dist': dist_matrix[idx1, idx2]
                    })
        else:
            min_dists.append(safe_distance * 2) # 只有一个点，绝对安全

        # 2. 速度检测 (动力学可行性)
        if prev_positions is not None:
            # 这里的减法是安全的，因为我们在开头做了 sort_values
            deltas = np.linalg.norm(positions - prev_positions, axis=1)
            vels = deltas / dt
            max_v = np.max(vels)
            max_vels.append(max_v)
        else:
            max_vels.append(0) # 第一帧速度为0

        prev_positions = positions

    # 生成控制台报告 
    print("\n" + "="*50)
    print("       🛡️ 无人机编队安全性验证报告 🛡️")
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
            print(f"  存在超速风险 (阈值 {max_velocity}m/s)")
        else: 
            print(f"  动力学安全")
        
        print(f"3. 碰撞事件: 共发现 {len(collision_events)} 帧次危险")
        if len(collision_events) > 0:
            print("   -> 危险样本 (前3条):")
            for i, event in enumerate(collision_events[:3]):
                print(f"      [T={event['Time']:.2f}s] {event['Drone1']} <-> {event['Drone2']} (距离: {event['Dist']:.3f}m)")
    else:
        print("数据异常：未检测到有效帧")

    print("="*50)

    # 绘制图表
    # 设置绘图风格
    plt.style.use('ggplot') 
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans'] # 适配中文
    plt.rcParams['axes.unicode_minus'] = False 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    time_steps = df['Time'].unique()
    
    # 绘制距离曲线
    ax1.plot(time_steps, min_dists, color='#2ca02c', linewidth=2, label='Min Distance')
    ax1.axhline(y=safe_distance, color='red', linestyle='--', linewidth=2, label=f'Safety Limit ({safe_distance}m)')
    ax1.set_ylabel('Distance (m)', fontsize=12)
    ax1.set_title('Swarm Safety Analysis: Minimum Separation Distance', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 填充危险区域
    ax1.fill_between(time_steps, 0, safe_distance, color='red', alpha=0.1)

    # 绘制速度曲线
    ax2.plot(time_steps, max_vels, color='#1f77b4', linewidth=2, label='Max Velocity')
    ax2.axhline(y=max_velocity, color='orange', linestyle='--', linewidth=2, label=f'Velocity Limit ({max_velocity}m/s)')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Swarm Dynamic Analysis: Maximum Velocity', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()
    print(f"图表已生成并保存至: {figure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='无人机编队安全分析工具')
    parser.add_argument('--csv', type=str, default=INPUT_CSV, help='CSV文件路径')
    parser.add_argument('--safe-distance', type=float, default=SAFE_DISTANCE, help='最小安全距离(米)')
    parser.add_argument('--max-velocity', type=float, default=MAX_VELOCITY, help='最大允许速度(米/秒)')
    parser.add_argument('--output', type=str, default=FIGURE_SAVE_PATH, help='图表保存路径')
    parser.add_argument('--interactive', action='store_true', help='交互式输入参数')
    
    args = parser.parse_args()
    
    analyze_safety(
        csv_file=args.csv,
        safe_distance=args.safe_distance,
        max_velocity=args.max_velocity,
        figure_path=args.output,
        interactive=args.interactive
    )