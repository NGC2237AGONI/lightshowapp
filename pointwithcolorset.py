import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def load_source_data(file_path="model_data.npz"):
    print(f"正在读取 {file_path} ...")
    try:
        data = np.load(file_path)
        return data['points'], data['colors']
    except FileNotFoundError:
        print("错误：找不到文件，生成测试数据...")
        # 生成一个测试用的长方体数据
        x = np.random.uniform(-10, 10, 10000)
        y = np.random.uniform(-2, 2, 10000) # Y轴很窄，模拟容易被拉伸的情况
        z = np.random.uniform(-5, 5, 10000)
        points = np.column_stack((x, y, z))
        colors = np.random.rand(10000, 3)
        return points, colors

def extract_outline_points_robust(points, colors, k=16, keep_percentage=5.0):
    """
    (保持原逻辑不变) 基于排名的特征提取
    """
    num_points = len(points)
    print(f"正在计算 {num_points} 个点的局部几何与颜色特征 (k={k})...")
    
    # 构建 KD-Tree
    tree = cKDTree(points)
    dist, indices = tree.query(points, k=k)
    
    # 1. 几何曲率
    neighbor_points = points[indices]
    centroid_neighbors = np.mean(neighbor_points, axis=1)
    geom_score = np.linalg.norm(points - centroid_neighbors, axis=1)
    
    # 2. 颜色边缘
    neighbor_colors = colors[indices]
    mean_neighbor_colors = np.mean(neighbor_colors, axis=1)
    color_score = np.linalg.norm(colors - mean_neighbor_colors, axis=1)
    
    # 3. 归一化
    if geom_score.max() > 0: geom_score /= geom_score.max()
    if color_score.max() > 0: color_score /= color_score.max()
        
    total_score = geom_score + color_score 

    # 4. 截断
    cutoff_value = np.percentile(total_score, 100 - keep_percentage)
    print(f"截断阈值 (Top {keep_percentage}%): {cutoff_value:.4f}")
    
    final_mask = total_score > cutoff_value
    
    return points[final_mask], colors[final_mask]

def visualize_result(pts, cols):
    if len(pts) == 0:
        print("错误：没有提取到任何点。")
        return

    fig = plt.figure(figsize=(10, 10)) # 画布设为正方形
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"正在渲染 {len(pts)} 个采样点...")
    
    # 绘制点
    # 注意：这里我们按照原始 x, y, z 绘制，不手动交换轴，
    # 依靠后面的 set_box_aspect 来控制视角
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=2.0, marker='.', alpha=0.9)
    
    # 黑色背景
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.axis('off')
    
    # ==========================================
    # 【核心修复】防止变形的逻辑
    # ==========================================
    # 1. 计算三个轴的中心点
    mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
    mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
    mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
    
    # 2. 找出最大的跨度 (Max Range)
    # 比如鱼身长 50米，宽 5米，高 10米，那 max_range 就是 50米
    range_x = pts[:, 0].max() - pts[:, 0].min()
    range_y = pts[:, 1].max() - pts[:, 1].min()
    range_z = pts[:, 2].max() - pts[:, 2].min()
    max_range = max(range_x, range_y, range_z) / 2.0
    
    # 3. 强行把 XYZ 的显示范围设为一样大
    # 这样 Matplotlib 就被迫使用 1:1:1 的真实比例，而不会自动拉伸短轴
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 4. 锁定 Box Aspect
    try:
        ax.set_box_aspect([1, 1, 1])
    except:
        pass
        
    # 5. 设置初始视角 (可选)
    # elev=0, azim=-90 通常是正面，你可以根据需要调整
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        raw_pts, raw_cols = load_source_data("model_data.npz")
        
        # 如果你的模型是 "侧躺" 的 (Y轴向上)，你需要在这里修正一下坐标
        # 将 Y 和 Z 互换，让它站起来
        # raw_pts = raw_pts[:, [0, 2, 1]] 
        
        sparse_pts, sparse_cols = extract_outline_points_robust(
            raw_pts, 
            raw_cols, 
            k=16,                
            keep_percentage=40.0  
        )
        
        visualize_result(sparse_pts, sparse_cols)
        
    except FileNotFoundError:
        print("错误：找不到 model_data.npz。")
    except Exception as e:
        print(f"发生错误: {e}")