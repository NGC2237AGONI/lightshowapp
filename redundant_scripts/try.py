import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ================= 配置 =================
AXIS_MODE = 1   # 0:原始, 1:Y/Z互换, 2:X/Z互换, ...
TARGET_COUNT = 800
SAFETY_DISTANCE = 1.5
# =======================================

def load_and_fix_data(file_path="model_data.npz", mode=0):
    print(f"正在读取 {file_path} ...")
    try:
        data = np.load(file_path)
        points = data['points']
        colors = data['colors']
        # 读取ID
        names = data['mesh_names']
        ids = data['vertex_ids']
        
        # 坐标轴变换
        if mode == 1: points = points[:, [0, 2, 1]]
        elif mode == 2: points = points[:, [2, 1, 0]]
        elif mode == 3: points = points[:, [1, 0, 2]]
        elif mode == 4: points[:, 1], points[:, 2] = points[:, 2].copy(), -points[:, 1].copy()
            
        # 归一化
        min_b = points.min(axis=0)
        max_b = points.max(axis=0)
        scale = np.max(max_b - min_b)
        if scale > 0: points = (points - min_b) / scale * 50.0 
        
        return points, colors, names, ids
    except Exception as e:
        print(f"错误：{e}")
        return [], [], [], []

def remove_noise_artifacts(points, colors, names, ids, connection_radius=3.0):
    """ 去噪"""
    if len(points) == 0: return points, colors, names, ids
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=connection_radius)
    if len(pairs) == 0: return points, colors, names, ids

    pairs = np.array(list(pairs))
    data = np.ones(len(pairs), dtype=bool)
    graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    n, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    largest_cluster_id = np.argmax(np.bincount(labels))
    mask = (labels == largest_cluster_id)
    
    print(f"去噪: 保留 {np.sum(mask)} / {len(points)}")
    return points[mask], colors[mask], names[mask], ids[mask]

def get_surface_voxel_sample(points, colors, names, ids, grid_size):
    """ 体素采样 (带身份追踪) """
    grid_indices = np.floor(points / grid_size).astype(int)
    _, unique_indices = np.unique(grid_indices, axis=0, return_index=True)
    # 按照索引提取
    return points[unique_indices], colors[unique_indices], names[unique_indices], ids[unique_indices]

def optimize_for_drone_count(points, colors, names, ids, target_count):
    print(f"目标: {target_count}...")
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    approx_area = 2 * np.sum((max_b - min_b) * np.roll(max_b - min_b, 1))
    
    current_grid_size = np.sqrt(approx_area / target_count) * 0.8 
    best_res = (points, colors, names, ids)
    min_error = float('inf')

    for i in range(20):
        # 这里的采样必须带上 names 和 ids
        t_pts, t_cols, t_nms, t_ids = get_surface_voxel_sample(points, colors, names, ids, current_grid_size)
        
        error = abs(len(t_pts) - target_count)
        if error < min_error:
            min_error = error
            best_res = (t_pts, t_cols, t_nms, t_ids)
            
        if error < (target_count * 0.05): break
        
        factor = np.clip(np.sqrt(len(t_pts) / target_count), 0.8, 1.2)
        current_grid_size *= factor
        
    return best_res

def finalize_drone_layout(curr_pts, curr_cols, curr_nms, curr_ids, 
                          src_pts, src_cols, src_nms, src_ids, 
                          target_count, min_dist=1.5):
    """ 补点/删点 (带身份追踪) """
    print("\n 最终检查与补全...")
    current_count = len(curr_pts)
    diff = target_count - current_count
    
    if diff == 0:
        return curr_pts, curr_cols, curr_nms, curr_ids
    
    elif diff < 0:
        print(f"随机移除 {abs(diff)} 个点")
        indices = np.random.choice(current_count, target_count, replace=False)
        return curr_pts[indices], curr_cols[indices], curr_nms[indices], curr_ids[indices]
    
    else:
        print(f"尝试补全 {diff} 个点...")
        source_tree = cKDTree(curr_pts)
        dists, _ = source_tree.query(src_pts, k=1)
        
        # 筛选合格的候选点
        mask = dists > min_dist
        cand_pts = src_pts[mask]
        cand_cols = src_cols[mask]
        cand_nms = src_nms[mask]
        cand_ids = src_ids[mask]
        cand_dists = dists[mask]
        
        num_to_add = min(len(cand_pts), diff)
        sorted_indices = np.argsort(cand_dists)[::-1][:num_to_add]
        
        # 合并
        final_pts = np.vstack((curr_pts, cand_pts[sorted_indices]))
        final_cols = np.vstack((curr_cols, cand_cols[sorted_indices]))
        final_nms = np.concatenate((curr_nms, cand_nms[sorted_indices]))
        final_ids = np.concatenate((curr_ids, cand_ids[sorted_indices]))
        
        return final_pts, final_cols, final_nms, final_ids

def visualize_final(pts, cols):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=15, alpha=0.9, edgecolors='none')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.axis('off')
    
    # 强制比例
    x_lim = [pts[:, 0].min(), pts[:, 0].max()]
    y_lim = [pts[:, 1].min(), pts[:, 1].max()]
    z_lim = [pts[:, 2].min(), pts[:, 2].max()]
    max_range = max([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]) / 2.0
    mid = [np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)]
    
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    try: ax.set_box_aspect([1, 1, 1])
    except: pass
    plt.show()

if __name__ == "__main__":

    pts, cols, nms, ids = load_and_fix_data("model_data.npz", mode=AXIS_MODE)
    
    if len(pts) > 0:

        cln_pts, cln_cols, cln_nms, cln_ids = remove_noise_artifacts(pts, cols, nms, ids, 3.0)
        

        opt_pts, opt_cols, opt_nms, opt_ids = optimize_for_drone_count(cln_pts, cln_cols, cln_nms, cln_ids, TARGET_COUNT)
        

        fin_pts, fin_cols, fin_nms, fin_ids = finalize_drone_layout(
            opt_pts, opt_cols, opt_nms, opt_ids,
            cln_pts, cln_cols, cln_nms, cln_ids,
            TARGET_COUNT, SAFETY_DISTANCE
        )
        

        output_file = "final_formation.npz"
        print(f"\n保存最终编队名单到 {output_file}...")
        print(f"  名单包含 {len(fin_nms)} 个点 (Mesh名 + 顶点ID)")
        np.savez(output_file, 
                 mesh_names=fin_nms, 
                 vertex_ids=fin_ids,
                 ref_points=fin_pts,
                 ref_colors=fin_cols)
        

        visualize_final(fin_pts, fin_cols)