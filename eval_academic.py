# --- START OF FILE eval_academic.py ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

def run_academic_evaluation():
    print("="*50)
    print(" 📊 无人机静态采样算法学术评测系统 📊")
    print("="*50)

    if not os.path.exists("model_data.npz") or not os.path.exists("final_formation.npz"):
        print("❌ 错误：找不到必要的 .npz 数据文件。请先在主程序中生成一次编队！")
        return

    # ================= 1. 加载底层数据 =================
    print("正在加载原始流形数据与无人机分布数据...")
    data_orig = np.load("model_data.npz")
    pts_orig = data_orig['points']
    cols_orig = data_orig['colors']

    data_form = np.load("final_formation.npz")
    drone_ids = data_form['vertex_ids']    # 无人机映射的原始顶点ID
    pts_final = data_form['ref_points']    # 经过防撞排斥后的最终物理坐标
    
    N_orig = len(pts_orig)
    N_drone = len(drone_ids)
    
    # 提取未经防撞推斥的无人机“原始采样坐标”（用于评估纯采样算法的几何误差）
    pts_sampled_raw = pts_orig[drone_ids]

    print(f"数据量: 原始模型 {N_orig} 点 | 采样无人机 {N_drone} 架")

    # ================= 2. 指标一：视觉显著性覆盖率 (Saliency Hit Rate) =================
    print("\n[1/3] 计算视觉显著性特征...")
    # 1. 计算原始模型每个点的视觉特征权重 (亮度 + 饱和度)
    cols_clip = np.clip(cols_orig, 0, 1)
    L = 0.299 * cols_clip[:,0] + 0.587 * cols_clip[:,1] + 0.114 * cols_clip[:,2]
    S = np.max(cols_clip, axis=1) - np.min(cols_clip, axis=1)
    Saliency = 0.1 + 0.5 * L + 0.4 * S
    Saliency = np.power(Saliency, 2)
    
    # 2. 划定“核心特征区” (例如，显著性排名前 20% 的点被认为是高光特征)
    threshold = np.percentile(Saliency, 80)
    
    # 3. 统计全模型的特征点分布
    total_feature_points = np.sum(Saliency >= threshold)
    
    # 4. 统计无人机落在“核心特征区”的数量
    drone_saliency = Saliency[drone_ids]
    drone_hit_feature = np.sum(drone_saliency >= threshold)
    hit_rate = drone_hit_feature / N_drone * 100
    
    print(f"  -> 定义全局前 20% 高亮区域为视觉焦点 (阈值 > {threshold:.3f})")
    print(f"  -> 无人机中属于高光特征区的数量: {drone_hit_feature} 架 ({hit_rate:.1f}%)")

    # ================= 3. 指标二：不对称倒角距离 (Asymmetric Chamfer Distance) =================
    print("\n[2/3] 计算几何外壳逼近倒角距离 (Chamfer Distance)...")
    # 测量：原模型表面的任意一点，距离最近的一架无人机有多远（评估“轮廓覆盖度”）
    tree_drones = cKDTree(pts_sampled_raw)
    dists, _ = tree_drones.query(pts_orig, k=1)
    cd_mean = np.mean(dists)
    
    print(f"  -> 平均倒角距离 (CD-Coverage): {cd_mean:.4f} 单位")

    # ================= 4. 指标三：物理安全距离分布 (Blue Noise NN-Dist) =================
    print("\n[3/3] 计算物理层防碰撞安全间距 (泊松盘蓝噪声特性)...")
    tree_final = cKDTree(pts_final)
    # 寻找每架飞机的最近邻距离（k=2，因为第一近的是自己距离为0）
    nn_dists, _ = tree_final.query(pts_final, k=2)
    nearest_dists = nn_dists[:, 1]
    
    min_dist = np.min(nearest_dists)
    mean_dist = np.mean(nearest_dists)
    print(f"  -> 最极限危险间距: {min_dist:.4f} 单位")
    print(f"  -> 集群平均近邻间距: {mean_dist:.4f} 单位")

    # ================= 5. 生成学术级图表 =================
    plt.style.use('ggplot')
    # 处理中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1：特征捕获率对比
    axs[0].bar(['全局平均预期 (随机)', '当前算法实际分配'], [20, hit_rate], color=['#7f7f7f', '#d62728'])
    axs[0].set_ylabel('特征点占比 (%)')
    axs[0].set_title('视觉显著性特征捕获率 (Saliency Hit Rate)')
    axs[0].text(0, 20, '20%', ha='center', va='bottom')
    axs[0].text(1, hit_rate, f'{hit_rate:.1f}%', ha='center', va='bottom')
    
    # 子图2：原始模型上的采样点分布直方图 (看看你的飞机主要集中在什么亮度的区域)
    axs[1].hist(Saliency, bins=50, alpha=0.5, density=True, label='全局像素池密度')
    axs[1].hist(drone_saliency, bins=50, alpha=0.7, density=True, label='采样机体分布')
    axs[1].axvline(threshold, color='r', linestyle='--', label='高光判定线')
    axs[1].set_xlabel('视觉显著性权重')
    axs[1].set_ylabel('概率密度')
    axs[1].set_title('无人机资源在视觉熵上的分配谱')
    axs[1].legend()

    # 子图3：最近邻距离分布 (泊松盘蓝噪声特性验证)
    axs[2].hist(nearest_dists, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    axs[2].axvline(min_dist, color='red', linestyle='--', label=f'极小间距: {min_dist:.2f}')
    axs[2].axvline(mean_dist, color='green', linestyle='-', label=f'均值: {mean_dist:.2f}')
    axs[2].set_xlabel('飞机间的最近距离 (单位)')
    axs[2].set_ylabel('无人机架数 (频数)')
    axs[2].set_title('最终成型的集群物理距离分布 (NN-Dist)')
    axs[2].legend()

    plt.tight_layout()
    chart_name = "Academic_Evaluation_Report.png"
    plt.savefig(chart_name, dpi=200)
    print("\n" + "="*50)
    print(f" 🎉 学术测评完成！详细多维分析图表已保存至: {chart_name}")
    print("="*50)

if __name__ == "__main__":
    run_academic_evaluation()
# --- END OF FILE ---