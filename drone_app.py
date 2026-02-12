import fbx
import sys
import csv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import texture_sampler  # 确保这个文件在同目录下

# ==========================================
# 模块 1: 数据提取 (DataExtractor)
# ==========================================
class DataExtractor:
    def __init__(self):
        # 初始化数据容器
        self.all_x = []
        self.all_y = []
        self.all_z = []
        self.all_c = []
        self.all_mesh_names = []
        self.all_vertex_ids = []

    def run(self, input_file, scale):
        """
        执行全量数据提取的主入口
        """
        # 每次运行前清空数据
        self.__init__()
        
        # 初始化 FBX 管理器
        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "Scene")
        importer = fbx.FbxImporter.Create(manager, "")

        # 加载文件
        if not importer.Initialize(input_file, -1, manager.GetIOSettings()):
            return False, f"错误: 无法打开文件 {input_file}"
        
        importer.Import(scene)
        importer.Destroy()

        # 开始遍历场景图
        root = scene.GetRootNode()
        if root:
            self._process_node(root, scale)

        manager.Destroy()

        # 检查提取结果
        if len(self.all_x) == 0:
            return False, "未提取到任何顶点数据，请检查模型格式。"

        # 整合数据为 Numpy 数组
        points = np.column_stack((self.all_x, self.all_y, self.all_z))
        colors = np.array(self.all_c)
        mesh_names = np.array(self.all_mesh_names)
        vertex_ids = np.array(self.all_vertex_ids)

        # 保存为中间文件
        np.savez("model_data.npz", 
                 points=points, 
                 colors=colors, 
                 mesh_names=mesh_names, 
                 vertex_ids=vertex_ids)
        
        return True, f"提取完成！共提取 {len(self.all_x)} 个点，已保存至 model_data.npz。"

    def _process_node(self, node, scale):
        """递归遍历节点树"""
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            self._extract_mesh_data(node, scale)
        
        for i in range(node.GetChildCount()):
            self._process_node(node.GetChild(i), scale)

    def _extract_mesh_data(self, node, scale):
        """提取单个 Mesh 的几何与颜色信息"""
        mesh = node.GetMesh()
        num_verts = mesh.GetControlPointsCount()
        mesh_name = node.GetName()
        
        # 1. 优先尝试读取贴图颜色 (通过 texture_sampler)
        colors = texture_sampler.get_texture_colors(mesh, node, base_path=".") 
        
        # 2. 如果没找到贴图，回退到属性读取 (顶点色/材质色)
        if colors is None:
            colors = self._get_vertex_colors(mesh, node)
        
        local_vertices = mesh.GetControlPoints()
        
        # 获取 t=0 时刻的全局变换矩阵
        time_zero = fbx.FbxTime(0)
        global_transform = node.EvaluateGlobalTransform(time_zero)
        
        for v_idx in range(num_verts):
            local_pos = local_vertices[v_idx]
            # 坐标变换：局部 -> 世界
            final_pos = global_transform.MultT(local_pos)
            
            self.all_x.append(final_pos[0] * scale)
            self.all_y.append(final_pos[1] * scale)
            self.all_z.append(final_pos[2] * scale)
            self.all_c.append(colors[v_idx])
            
            # 记录身份信息
            self.all_mesh_names.append(mesh_name)
            self.all_vertex_ids.append(v_idx)

    def _get_vertex_colors(self, mesh, node):
        """备用颜色获取逻辑"""
        num_verts = mesh.GetControlPointsCount()
        default_color = (0.5, 0.5, 0.5) 
        final_colors = [default_color] * num_verts 
        
        # 尝试顶点色
        vertex_color_layer = mesh.GetElementVertexColor(0)
        if vertex_color_layer:
            direct_array = vertex_color_layer.GetDirectArray()
            if direct_array.GetCount() > 0:
                for i in range(num_verts):
                    idx = i if i < direct_array.GetCount() else 0
                    c = direct_array.GetAt(idx)
                    final_colors[i] = (c[0], c[1], c[2])
                return final_colors
        
        return final_colors

# ==========================================
# 模块 2: 编队优化 (FormationOptimizer)
# ==========================================
class FormationOptimizer:
    def run(self, axis_mode, target_count, safety_distance):
        """
        执行编队优化：去噪 -> 采样 -> 补漏
        """
        try:
            # 1. 加载并修正坐标轴
            pts, cols, nms, ids = self._load_and_fix_data("model_data.npz", axis_mode)
            if len(pts) == 0: 
                return False, "加载 model_data.npz 失败，请先执行提取步骤。", None, None

            # 2. 连通域去噪
            cln_pts, cln_cols, cln_nms, cln_ids = self._remove_noise_artifacts(pts, cols, nms, ids, 3.0)
            
            # 3. 自适应体素采样 (核心筛选)
            opt_pts, opt_cols, opt_nms, opt_ids = self._optimize_for_drone_count(cln_pts, cln_cols, cln_nms, cln_ids, target_count)
            
            # 4. 最终补漏与检查
            fin_pts, fin_cols, fin_nms, fin_ids = self._finalize_drone_layout(
                opt_pts, opt_cols, opt_nms, opt_ids,
                cln_pts, cln_cols, cln_nms, cln_ids,
                target_count, safety_distance
            )

            # 保存最终结果到 final_formation.npz
            np.savez("final_formation.npz", 
                     mesh_names=fin_nms, 
                     vertex_ids=fin_ids,
                     ref_points=fin_pts, 
                     ref_colors=fin_cols)
            
            return True, f"优化完成，最终点数: {len(fin_pts)} (目标: {target_count})", fin_pts, fin_cols
            
        except Exception as e:
            return False, f"优化过程中出错: {str(e)}", None, None

    def _load_and_fix_data(self, file_path, mode):
        try:
            data = np.load(file_path)
            points = data['points']
            colors = data['colors']
            names = data['mesh_names']
            ids = data['vertex_ids']
            
            # 详细的坐标轴变换逻辑
            if mode == 1: 
                points = points[:, [0, 2, 1]] # Y/Z 互换
            elif mode == 2: 
                points = points[:, [2, 1, 0]] # X/Z 互换
            elif mode == 3: 
                points = points[:, [1, 0, 2]] # X/Y 互换
            elif mode == 4: 
                # 绕 X 轴旋转 90 度
                points[:, 1], points[:, 2] = points[:, 2].copy(), -points[:, 1].copy()
                
            # 归一化处理
            min_b = points.min(axis=0)
            max_b = points.max(axis=0)
            scale = np.max(max_b - min_b)
            if scale > 0: 
                points = (points - min_b) / scale * 50.0 
            
            return points, colors, names, ids
        except: 
            return [], [], [], []

    def _remove_noise_artifacts(self, points, colors, names, ids, connection_radius=3.0):
        if len(points) == 0: return points, colors, names, ids
        
        # 建立 KDTree
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=connection_radius)
        if len(pairs) == 0: return points, colors, names, ids

        # 连通域分析
        pairs = np.array(list(pairs))
        data = np.ones(len(pairs), dtype=bool)
        graph = csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
        n, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        # 保留最大连通域
        largest_cluster_id = np.argmax(np.bincount(labels))
        mask = (labels == largest_cluster_id)
        
        return points[mask], colors[mask], names[mask], ids[mask]

    def _get_surface_voxel_sample(self, points, colors, names, ids, grid_size):
        # 体素化网格采样
        grid_indices = np.floor(points / grid_size).astype(int)
        _, unique_indices = np.unique(grid_indices, axis=0, return_index=True)
        return points[unique_indices], colors[unique_indices], names[unique_indices], ids[unique_indices]

    def _optimize_for_drone_count(self, points, colors, names, ids, target_count):
        # 估算初始网格大小
        min_b = points.min(axis=0)
        max_b = points.max(axis=0)
        approx_area = 2 * np.sum((max_b - min_b) * np.roll(max_b - min_b, 1))
        current_grid_size = np.sqrt(approx_area / target_count) * 0.8 
        
        best_res = (points, colors, names, ids)
        min_error = float('inf')

        # 迭代逼近目标数量
        for i in range(20):
            t_pts, t_cols, t_nms, t_ids = self._get_surface_voxel_sample(points, colors, names, ids, current_grid_size)
            
            error = abs(len(t_pts) - target_count)
            if error < min_error:
                min_error = error
                best_res = (t_pts, t_cols, t_nms, t_ids)
                
            if error < (target_count * 0.05): 
                break
            
            # 调整网格大小
            factor = np.clip(np.sqrt(len(t_pts) / target_count), 0.8, 1.2)
            current_grid_size *= factor
            
        return best_res

    def _finalize_drone_layout(self, curr_pts, curr_cols, curr_nms, curr_ids, src_pts, src_cols, src_nms, src_ids, target_count, min_dist):
        current_count = len(curr_pts)
        diff = target_count - current_count
        
        if diff == 0: 
            return curr_pts, curr_cols, curr_nms, curr_ids
        
        elif diff < 0:
            # 随机移除多余点
            indices = np.random.choice(current_count, target_count, replace=False)
            return curr_pts[indices], curr_cols[indices], curr_nms[indices], curr_ids[indices]
        
        else:
            # 智能补点：从源数据中寻找离现有集最远的点
            source_tree = cKDTree(curr_pts)
            dists, _ = source_tree.query(src_pts, k=1)
            
            # 筛选满足安全距离的候选点
            mask = dists > min_dist
            cand_pts = src_pts[mask]
            cand_cols = src_cols[mask]
            cand_nms = src_nms[mask]
            cand_ids = src_ids[mask]
            cand_dists = dists[mask]
            
            num_to_add = min(len(cand_pts), diff)
            if num_to_add == 0: 
                return curr_pts, curr_cols, curr_nms, curr_ids
            
            # 优先选距离最远的
            sorted_indices = np.argsort(cand_dists)[::-1][:num_to_add]
            
            return (np.vstack((curr_pts, cand_pts[sorted_indices])),
                    np.vstack((curr_cols, cand_cols[sorted_indices])),
                    np.concatenate((curr_nms, cand_nms[sorted_indices])),
                    np.concatenate((curr_ids, cand_ids[sorted_indices])))

# ==========================================
# 模块 3: 动画导出 (AnimationExporter)
# ==========================================
class AnimationExporter:
    def __init__(self):
        self.TARGET_MAP = {}
        self.COLOR_MAP = {}
        self.SKINNING_DATA = {}

    def get_animations(self, fbx_file):
        """扫描 FBX 文件获取动画列表"""
        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "Scene")
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(fbx_file, -1, manager.GetIOSettings()): return []
        importer.Import(scene)
        importer.Destroy()
        
        criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        num_stacks = scene.GetSrcObjectCount(criteria)
        anims = []
        for i in range(num_stacks):
            s = scene.GetSrcObject(criteria, i)
            span = s.GetLocalTimeSpan()
            dur = span.GetStop().GetSecondDouble() - span.GetStart().GetSecondDouble()
            anims.append({"name": s.GetName(), "duration": dur, "index": i})
        manager.Destroy()
        return anims

    def run_export(self, fbx_file, anim_index, custom_duration, fps, scale, axis_mode, static_threshold, end_buffer, boundaries):
        """
        核心导出函数
        参数包含：边界限制、时间缩放、骨骼解算等
        """
        # 1. 加载 final_formation.npz
        try:
            data = np.load("final_formation.npz")
            names = data['mesh_names']
            ids = data['vertex_ids']
            colors = data['ref_colors']
            self.TARGET_MAP = {}
            self.COLOR_MAP = {}
            for n, i, c in zip(names, ids, colors):
                n_str = str(n)
                key = (n_str, int(i))
                if n_str not in self.TARGET_MAP: self.TARGET_MAP[n_str] = set()
                self.TARGET_MAP[n_str].add(int(i))
                self.COLOR_MAP[key] = c
        except: return False, "找不到 final_formation.npz"

        # 2. 准备 FBX
        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "Scene")
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(fbx_file, -1, manager.GetIOSettings()): return False, "无法读取FBX"
        importer.Import(scene)
        importer.Destroy()

        # 3. 设置动画
        criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        target_stack = scene.GetSrcObject(criteria, anim_index)
        scene.SetCurrentAnimationStack(target_stack)
        
        span = target_stack.GetLocalTimeSpan()
        start = span.GetStart().GetSecondDouble()
        
        # 计算最大帧
        max_frames = int(custom_duration * fps)

        # 4. 预计算骨骼蒙皮数据
        self._prepare_skinning_data(scene)

        # 5. 导出 CSV
        try:
            with open("drone_path.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
                root = scene.GetRootNode()
                if root:
                    # 传入 boundaries 参数
                    self._process_node_smart(root, start, max_frames, fps, scale, axis_mode, static_threshold, end_buffer, boundaries, writer)
        except Exception as e:
            return False, f"导出失败: {str(e)}"
        
        manager.Destroy()
        return True, "导出成功！"

    def _prepare_skinning_data(self, scene):
        """ 修复版骨骼数据读取逻辑 (兼容性增强) """
        self.SKINNING_DATA = {} 
        src_len = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId))
        for i in range(src_len):
            mesh = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId), i)
            node = mesh.GetNode()
            if not node: continue
            mesh_name = node.GetName()
            if mesh_name not in self.TARGET_MAP: continue
            target_indices = self.TARGET_MAP[mesh_name]
            
            skin_deformer = None
            for j in range(mesh.GetDeformerCount()):
                df = mesh.GetDeformer(j)
                # 使用 GetClassId 兼容性判定
                if df.GetClassId() == fbx.FbxSkin.ClassId:
                    skin_deformer = df
                    break
            if not skin_deformer: continue 
            
            skin = skin_deformer 
            try: cluster_count = skin.GetClusterCount()
            except: continue

            for c_idx in range(cluster_count):
                cluster = skin.GetCluster(c_idx)
                bone_node = cluster.GetLink() 
                if not bone_node: continue
                lMatrix = fbx.FbxAMatrix()
                cluster.GetTransformLinkMatrix(lMatrix)
                bind_matrix_inv = lMatrix.Inverse()
                indices = cluster.GetControlPointIndices()
                weights = cluster.GetControlPointWeights()
                for k in range(cluster.GetControlPointIndicesCount()):
                    v_idx = indices[k]
                    w = weights[k]
                    if v_idx in target_indices:
                        key = (mesh_name, v_idx)
                        if key not in self.SKINNING_DATA: self.SKINNING_DATA[key] = []
                        self.SKINNING_DATA[key].append( (bone_node, bind_matrix_inv, w) )

    def _process_node_smart(self, node, start_time, max_frames, fps, scale, axis_mode, static_thresh, end_buffer, boundaries, writer):
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            mesh_name = node.GetName()
            if mesh_name in self.TARGET_MAP:
                self._extract_and_prune_data(node, mesh_name, start_time, max_frames, fps, scale, axis_mode, static_thresh, end_buffer, boundaries, writer)
        for i in range(node.GetChildCount()):
            self._process_node_smart(node.GetChild(i), start_time, max_frames, fps, scale, axis_mode, static_thresh, end_buffer, boundaries, writer)

    def _extract_and_prune_data(self, node, mesh_name, start_time, max_frames, fps, scale, axis_mode, static_thresh, end_buffer, boundaries, writer):
        """
        智能轨迹计算：
        1. 骨骼/刚体解算
        2. 空间钳制 (Boundaries)
        3. 智能静止检测与剪辑
        """
        mesh = node.GetMesh()
        target_ids = self.TARGET_MAP[mesh_name]
        local_verts = mesh.GetControlPoints()
        fbx_time = fbx.FbxTime()
        has_skin = (mesh_name, list(target_ids)[0]) in self.SKINNING_DATA

        frame_buffer = [] 
        prev_frame_coords = None
        last_moving_frame = 0 
        
        # 解包边界
        min_x, max_x, min_y, max_y, min_z, max_z = boundaries

        for f in range(max_frames + 1):
            curr_sec = start_time + (f / fps)
            fbx_time.SetSecondDouble(curr_sec)
            global_trans = node.EvaluateGlobalTransform(fbx_time)
            
            current_frame_data = [] 
            current_coords_only = [] 

            for v_idx in target_ids:
                local_pos = local_verts[v_idx]
                final_vec = fbx.FbxVector4(0, 0, 0, 0)
                
                if has_skin:
                    skin_info = self.SKINNING_DATA.get((mesh_name, v_idx), [])
                    if not skin_info: final_vec = global_trans.MultT(local_pos)
                    else:
                        vertex_accumulated = fbx.FbxVector4(0, 0, 0, 0)
                        for bone_node, bind_inv, weight in skin_info:
                            curr_bone_matrix = bone_node.EvaluateGlobalTransform(fbx_time)
                            deform_matrix = curr_bone_matrix * bind_inv
                            influenced_pos = deform_matrix.MultT(local_pos)
                            vertex_accumulated += influenced_pos * weight
                        final_vec = vertex_accumulated
                else:
                    final_vec = global_trans.MultT(local_pos)
                
                x = final_vec[0] * scale
                y = final_vec[1] * scale
                z = final_vec[2] * scale
                
                if axis_mode == 1: x, y, z = x, z, y
                elif axis_mode == 2: x, y, z = z, y, x
                elif axis_mode == 3: x, y, z = y, x, z
                elif axis_mode == 4: y, z = z, -y
                
                # [核心功能] 空间钳制
                x = np.clip(x, min_x, max_x)
                y = np.clip(y, min_y, max_y)
                z = np.clip(z, min_z, max_z)
                
                rgb = self.COLOR_MAP.get((mesh_name, v_idx), [1.0, 1.0, 1.0])
                r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
                
                current_frame_data.append( (v_idx, x, y, z, f/fps, r, g, b) )
                current_coords_only.append( [x, y, z] )
            
            frame_buffer.append(current_frame_data)
            
            # 智能静止检测逻辑
            curr_coords_np = np.array(current_coords_only)
            if prev_frame_coords is not None:
                movements = np.linalg.norm(curr_coords_np - prev_frame_coords, axis=1)
                if np.max(movements) > static_thresh: last_moving_frame = f
            prev_frame_coords = curr_coords_np

        buffer_frames = int(end_buffer * fps)
        final_end_frame = min(last_moving_frame + buffer_frames, max_frames)
        final_end_frame = max(final_end_frame, 1) 
        
        for f in range(final_end_frame):
            for item in frame_buffer[f]:
                # 写入 [Frame, Time, Obj, ID, X, Y, Z, R, G, B]
                v_idx, x, y, z, sec, r, g, b = item
                writer.writerow([f, f"{sec:.3f}", mesh_name, v_idx, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", r, g, b])

# ==========================================
# 模块 4: 轨迹自动优化器 (TrajectoryOptimizer)
# ==========================================
class TrajectoryOptimizer:
    def optimize_trajectory(self, csv_file, safe_dist, max_vel, bound_L, bound_W, bound_H, manual_time_scale=None):
        try:
            df = pd.read_csv(csv_file)
            df = df.sort_values(by=['Frame', 'Object', 'VertexID'])
            
            # --- 1. 空间缩放 (基于安全距离) ---
            frames = df['Frame'].unique()
            min_distances = []
            sample_step = max(1, len(frames) // 20)
            
            for f in frames[::sample_step]:
                current_data = df[df['Frame'] == f]
                pos = current_data[['X', 'Y', 'Z']].values
                if len(pos) > 1:
                    dists = pdist(pos)
                    if len(dists) > 0: min_distances.append(np.min(dists))
            
            min_dist_original = np.min(min_distances) if min_distances else 0.001
            if min_dist_original <= 0: min_dist_original = 0.001
            
            spatial_scale = 1.0
            if min_dist_original < safe_dist:
                spatial_scale = safe_dist / min_dist_original * 1.05 # 稍微留点余量
            
            # 边界限制检查
            all_pos = df[['X', 'Y', 'Z']].values
            min_xyz = all_pos.min(axis=0)
            max_xyz = all_pos.max(axis=0)
            dims = max_xyz - min_xyz
            pred_dims = dims * spatial_scale
            
            limit_x = bound_L / pred_dims[0] if pred_dims[0] > bound_L else 999
            limit_y = bound_W / pred_dims[1] if pred_dims[1] > bound_W else 999
            limit_z = bound_H / pred_dims[2] if pred_dims[2] > bound_H else 999
            limit_scale = min(limit_x, limit_y, limit_z)
            
            if limit_scale < 1.0:
                spatial_scale *= limit_scale
                
            # 应用空间缩放
            center = (max_xyz + min_xyz) / 2
            df['X'] = center[0] + (df['X'] - center[0]) * spatial_scale
            df['Y'] = center[1] + (df['Y'] - center[1]) * spatial_scale
            df['Z'] = center[2] + (df['Z'] - center[2]) * spatial_scale
            
            # --- 2. 时间缩放 (基于最大速度) ---
            df.sort_values(['VertexID', 'Frame'], inplace=True)
            dt = df['Time'].diff().mean()
            if dt <= 0: dt = 0.05
            
            df['dX'] = df['X'].diff()
            df['dY'] = df['Y'].diff()
            df['dZ'] = df['Z'].diff()
            df['ID_diff'] = df['VertexID'].diff()
            valid_moves = df[df['ID_diff'] == 0]
            
            dist_sq = valid_moves['dX']**2 + valid_moves['dY']**2 + valid_moves['dZ']**2
            max_dist_step = np.sqrt(dist_sq.max())
            curr_max_vel = max_dist_step / dt
            
            time_scale = 1.0
            if curr_max_vel > max_vel:
                time_scale = curr_max_vel / max_vel * 1.1
            
            if manual_time_scale:
                time_scale = manual_time_scale
            
            df['Time'] *= time_scale
            
            # 保存
            output_file = "drone_path_optimized.csv"
            cols = ["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"]
            df[cols].to_csv(output_file, index=False)
            
            info = {
                'spatial_scale': spatial_scale,
                'time_scale': time_scale,
                'orig_min_dist': min_dist_original,
                'final_max_vel': curr_max_vel / time_scale
            }
            return True, "优化完成", output_file, info
            
        except Exception as e:
            return False, f"优化失败: {str(e)}", "", {}