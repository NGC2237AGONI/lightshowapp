import fbx
import sys
import os 
import csv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter 
import texture_sampler  
import traceback

# ==========================================
# 模块 1: 数据提取 (DataExtractor)
# ==========================================
class DataExtractor:
    def __init__(self):
        self.all_x = []
        self.all_y = []
        self.all_z = []
        self.all_c = []
        self.all_mesh_names = []
        self.all_vertex_ids = []
        self.fbx_dir = "." 

    def run(self, input_file, scale):
        self.__init__()
        self.fbx_dir = os.path.dirname(os.path.abspath(input_file))
        
        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "Scene")
        importer = fbx.FbxImporter.Create(manager, "")

        if not importer.Initialize(input_file, -1, manager.GetIOSettings()):
            return False, f"错误: 无法打开文件 {input_file}"
        importer.Import(scene)
        importer.Destroy()

        root = scene.GetRootNode()
        if root:
            self._process_node(root, scale)

        manager.Destroy()

        if len(self.all_x) == 0:
            return False, "未提取到数据"

        points = np.column_stack((self.all_x, self.all_y, self.all_z))
        colors = np.array(self.all_c)
        
        # 【增强版颜色保底】
        # 如果颜色标准差很小（说明颜色单一，比如全黑或全灰），强制应用彩虹色
        if np.std(colors) < 0.05:
            print("   [提示] 检测到模型颜色单一（贴图可能丢失），强制应用'彩虹高度图'保底。")
            z_vals = points[:, 2]
            if len(z_vals) > 0:
                z_min, z_max = z_vals.min(), z_vals.max()
                z_range = z_max - z_min + 0.001
                for i in range(len(colors)):
                    h = (z_vals[i] - z_min) / z_range
                    # 简单的热力图映射 (蓝->红)
                    r = max(0, min(1, 2 * h - 0.5))
                    g = max(0, min(1, 1 - 2 * abs(h - 0.5)))
                    b = max(0, min(1, 1 - 2 * h))
                    colors[i] = [r, g, b]

        mesh_names = np.array(self.all_mesh_names)
        vertex_ids = np.array(self.all_vertex_ids)

        np.savez("model_data.npz", 
                 points=points, colors=colors, 
                 mesh_names=mesh_names, vertex_ids=vertex_ids)
        
        return True, f"提取完成！原始采样池: {len(self.all_x)} 点 (高画质模式)"

    def _process_node(self, node, scale):
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            self._extract_mesh_data(node, scale)
        for i in range(node.GetChildCount()):
            self._process_node(node.GetChild(i), scale)

    def _extract_mesh_data(self, node, scale):
        mesh = node.GetMesh()
        num_verts = mesh.GetControlPointsCount()
        mesh_name = node.GetName()
        
        target_quota = 8000 
        if num_verts <= target_quota:
            indices_to_take = range(num_verts)
        else:
            indices_to_take = np.linspace(0, num_verts-1, target_quota, dtype=int)
            indices_to_take = np.unique(indices_to_take)
        
        colors = texture_sampler.get_texture_colors(mesh, node, base_path=self.fbx_dir) 
        if colors is None:
            colors = self._get_vertex_colors(mesh, node)
        
        local_vertices = mesh.GetControlPoints()
        time_zero = fbx.FbxTime(0)
        global_transform = node.EvaluateGlobalTransform(time_zero)
        
        for v_idx in indices_to_take:
            local_pos = local_vertices[v_idx]
            final_pos = global_transform.MultT(local_pos)
            
            self.all_x.append(final_pos[0] * scale)
            self.all_y.append(final_pos[1] * scale)
            self.all_z.append(final_pos[2] * scale)
            self.all_c.append(colors[v_idx])
            self.all_mesh_names.append(mesh_name)
            self.all_vertex_ids.append(v_idx) 

    def _get_vertex_colors(self, mesh, node):
        num_verts = mesh.GetControlPointsCount()
        # 默认暗灰色，方便 run 方法里识别并替换为彩虹色
        default_color = (0.2, 0.2, 0.2) 
        final_colors = [default_color] * num_verts 
        
        vertex_color_layer = mesh.GetElementVertexColor(0)
        if vertex_color_layer:
            direct_array = vertex_color_layer.GetDirectArray()
            index_array = vertex_color_layer.GetIndexArray()
            mapping_mode = vertex_color_layer.GetMappingMode()
            ref_mode = vertex_color_layer.GetReferenceMode()

            if direct_array.GetCount() > 0:
                for i in range(num_verts):
                    color_idx = 0
                    if mapping_mode == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
                        if ref_mode == fbx.FbxLayerElement.EReferenceMode.eDirect: color_idx = i
                        elif ref_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect: color_idx = index_array.GetAt(i)
                    elif mapping_mode == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
                        if ref_mode == fbx.FbxLayerElement.EReferenceMode.eDirect: color_idx = i
                        elif ref_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect: color_idx = i % direct_array.GetCount() 
                    
                    if color_idx >= direct_array.GetCount(): color_idx = 0
                    c = direct_array.GetAt(color_idx)
                    try: final_colors[i] = (c.mRed, c.mGreen, c.mBlue)
                    except: 
                        try: final_colors[i] = (c[0], c[1], c[2])
                        except: final_colors[i] = (0.2, 0.2, 0.2)
                return final_colors
        return final_colors

# ==========================================
# 模块 2: 编队优化 (FormationOptimizer)
# ==========================================
class FormationOptimizer:
    def run(self, axis_mode, target_count, safety_distance):
        try:
            pts, cols, nms, ids = self._load_and_fix_data("model_data.npz", axis_mode)
            if len(pts) == 0: return False, "加载失败", None, None

            HARD_LIMIT = 8000 
            if len(pts) > HARD_LIMIT:
                print(f"   [V4.6] 触发总数熔断: {len(pts)} -> {HARD_LIMIT}")
                indices = np.linspace(0, len(pts)-1, HARD_LIMIT, dtype=int)
                indices = np.unique(indices)
                pts = pts[indices]
                cols = cols[indices]
                nms = nms[indices]
                ids = ids[indices]
            
            if not np.isfinite(pts).all(): pts = np.nan_to_num(pts, nan=0.0, posinf=100.0, neginf=-100.0)
            center = np.mean(pts, axis=0)
            pts_centered = pts - center
            max_range = np.max(np.abs(pts_centered))
            if max_range < 1e-5: max_range = 1.0
            scale_factor = 50.0 / max_range 
            pts_norm = pts_centered * scale_factor
            
            estimated_vol = (100.0 ** 3) * 0.2
            grid_size = (estimated_vol / target_count) ** (1/3)
            opt_indices = self._robust_voxel_sample(pts_norm, grid_size, target_count)
            
            fin_pts = pts[opt_indices]
            fin_cols = cols[opt_indices]
            fin_nms = nms[opt_indices]
            fin_ids = ids[opt_indices]
            
            diff = target_count - len(fin_pts)
            if diff > 0:
                all_idx = set(range(len(pts)))
                sel_idx = set(opt_indices)
                rem_idx = list(all_idx - sel_idx)
                if rem_idx:
                    add_idx = np.random.choice(rem_idx, min(diff, len(rem_idx)), replace=False)
                    fin_pts = np.vstack((fin_pts, pts[add_idx]))
                    fin_cols = np.vstack((fin_cols, cols[add_idx]))
                    fin_nms = np.concatenate((fin_nms, nms[add_idx]))
                    fin_ids = np.concatenate((fin_ids, ids[add_idx]))
            
            fin_pts = self._pre_relax(fin_pts, safety_distance * 0.8)
            fin_pts = fin_pts - np.mean(fin_pts, axis=0)

            np.savez("final_formation.npz", 
                     mesh_names=fin_nms, vertex_ids=fin_ids,
                     ref_points=fin_pts, ref_colors=fin_cols)
            
            return True, f"优化完成! 选中 {len(fin_pts)} 点 (高画质)", fin_pts, fin_cols
        except Exception as e:
            traceback.print_exc()
            return False, f"优化出错: {str(e)}", None, None

    def _pre_relax(self, points, min_dist):
        if len(points) == 0: return points
        for _ in range(10): 
            tree = cKDTree(points)
            pairs = tree.query_pairs(r=min_dist)
            if not pairs: break
            for i, j in pairs:
                vec = points[i] - points[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-4: vec = np.random.rand(3) * 0.01; dist = 0.01
                push = (min_dist - dist) * 0.5 * (vec / dist)
                points[i] += push
                points[j] -= push
        return points

    def _robust_voxel_sample(self, points, start_grid, target):
        best_indices = np.arange(len(points))
        min_err = float('inf')
        grid = start_grid
        for i in range(10): 
            grid_idx = np.floor(points / (grid + 1e-6)).astype(int)
            _, uniq_idx = np.unique(grid_idx, axis=0, return_index=True)
            count = len(uniq_idx)
            err = abs(count - target)
            if err < min_err:
                min_err = err; best_indices = uniq_idx
            if err < target * 0.1: break
            ratio = (count / target) ** (1/3)
            ratio = np.clip(ratio, 0.5, 1.5)
            grid *= ratio
        return best_indices

    def _load_and_fix_data(self, file_path, mode):
        try:
            data = np.load(file_path)
            points = data['points']; colors = data['colors']; names = data['mesh_names']; ids = data['vertex_ids']
            if mode == 1: points = points[:, [0, 2, 1]]
            elif mode == 2: points = points[:, [2, 1, 0]]
            elif mode == 3: points = points[:, [1, 0, 2]]
            elif mode == 4: points[:, 1], points[:, 2] = points[:, 2].copy(), -points[:, 1].copy()
            return points, colors, names, ids
        except: return [], [], [], []

# ==========================================
# 模块 3: 动画导出 (AnimationExporter)
# ==========================================
class AnimationExporter:
    def __init__(self):
        self.TARGET_MAP = {}
        self.COLOR_MAP = {}
        self.SKINNING_DATA = {}

    def get_animations(self, fbx_file):
        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "Scene")
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(fbx_file, -1, manager.GetIOSettings()): return []
        importer.Import(scene); importer.Destroy(); criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        num = scene.GetSrcObjectCount(criteria); anims = []
        for i in range(num):
            s = scene.GetSrcObject(criteria, i); span = s.GetLocalTimeSpan(); dur = span.GetStop().GetSecondDouble() - span.GetStart().GetSecondDouble()
            anims.append({"name": s.GetName(), "duration": dur, "index": i})
        manager.Destroy(); return anims

    # 【新增】原始导出方法，解决 AttributeError
    def run_raw_export(self, fbx_file, anim_index, fps, scale, axis_mode, output_path):
        try:
            data = np.load("final_formation.npz")
            names = data['mesh_names']; ids = data['vertex_ids']; colors = data['ref_colors']
            self.TARGET_MAP = {}; self.COLOR_MAP = {}
            for n, i, c in zip(names, ids, colors):
                key = (str(n), int(i)); self.TARGET_MAP.setdefault(str(n), set()).add(int(i)); self.COLOR_MAP[key] = c
        except: return False, "No npz"

        manager = fbx.FbxManager.Create(); scene = fbx.FbxScene.Create(manager, "Scene"); importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(fbx_file, -1, manager.GetIOSettings()): return False, "Err"
        importer.Import(scene); importer.Destroy()

        criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        target_stack = scene.GetSrcObject(criteria, anim_index)
        scene.SetCurrentAnimationStack(target_stack)
        
        span = target_stack.GetLocalTimeSpan()
        start = span.GetStart().GetSecondDouble()
        end = span.GetStop().GetSecondDouble()
        
        # 导出全时长
        total_time = end - start
        max_frames = int(total_time * fps)
        self._prepare_skinning_data(scene)

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
                root = scene.GetRootNode()
                if root:
                    self._process_node_raw(root, start, max_frames, fps, scale, axis_mode, writer)
        except Exception as e: return False, str(e)
        
        manager.Destroy(); return True, "Success"

    def _prepare_skinning_data(self, scene):
        self.SKINNING_DATA = {} 
        src_len = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId))
        for i in range(src_len):
            mesh = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId), i); node = mesh.GetNode()
            if not node or node.GetName() not in self.TARGET_MAP: continue
            m_name = node.GetName(); target_ids = self.TARGET_MAP[m_name]
            skin_deformer = None
            for j in range(mesh.GetDeformerCount()):
                if mesh.GetDeformer(j).GetClassId() == fbx.FbxSkin.ClassId: skin_deformer = mesh.GetDeformer(j); break
            if not skin_deformer: continue 
            for c_idx in range(skin_deformer.GetClusterCount()):
                cluster = skin_deformer.GetCluster(c_idx); bone = cluster.GetLink()
                if not bone: continue
                lMatrix = fbx.FbxAMatrix(); cluster.GetTransformLinkMatrix(lMatrix); bind_inv = lMatrix.Inverse()
                ind = cluster.GetControlPointIndices(); wht = cluster.GetControlPointWeights()
                for k in range(cluster.GetControlPointIndicesCount()):
                    v_idx = ind[k]
                    if v_idx in target_ids:
                        self.SKINNING_DATA.setdefault((m_name, v_idx), []).append((bone, bind_inv, wht[k]))

    def _process_node_raw(self, node, start, frames, fps, scale, mode, writer):
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            if node.GetName() in self.TARGET_MAP:
                self._extract_data_raw(node, start, frames, fps, scale, mode, writer)
        for i in range(node.GetChildCount()):
            self._process_node_raw(node.GetChild(i), start, frames, fps, scale, mode, writer)

    def _extract_data_raw(self, node, start, frames, fps, scale, mode, writer):
        mesh = node.GetMesh(); m_name = node.GetName(); target_ids = self.TARGET_MAP[m_name]
        l_verts = mesh.GetControlPoints(); t = fbx.FbxTime()
        for f in range(frames + 1):
            curr_time = f / fps
            t.SetSecondDouble(start + curr_time)
            
            g_trans = node.EvaluateGlobalTransform(t)
            for v_idx in target_ids:
                final = fbx.FbxVector4(0,0,0,0)
                if (m_name, v_idx) in self.SKINNING_DATA:
                    for bone, binv, w in self.SKINNING_DATA[(m_name, v_idx)]:
                        final += (bone.EvaluateGlobalTransform(t) * binv).MultT(l_verts[v_idx]) * w
                else: final = g_trans.MultT(l_verts[v_idx])
                x, y, z = final[0]*scale, final[1]*scale, final[2]*scale
                if mode == 1: x,y,z = x,z,y
                elif mode == 2: x,y,z = z,y,x
                elif mode == 3: x,y,z = y,x,z
                elif mode == 4: y,z = z,-y
                rgb = self.COLOR_MAP.get((m_name, v_idx), [1,1,1])
                writer.writerow([f, f"{curr_time:.3f}", m_name, v_idx, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)])

# ==========================================
# 模块 4: 轨迹自动优化器 (TrajectoryOptimizer)
# ==========================================
class TrajectoryOptimizer:
    # 【新增】智能裁剪与循环逻辑
    def smart_trim_and_loop(self, input_csv, output_csv, loop_count):
        try:
            df = pd.read_csv(input_csv)
            if df.empty: return False, "数据为空"
            
            # 1. 忽略最后 0.5s 防止循环瞬移干扰
            max_time = df['Time'].max()
            safe_end = max_time - 0.5
            if safe_end < 0.5: safe_end = max_time
            
            df_safe = df[df['Time'] <= safe_end]
            
            # 2. 计算每一帧的整体运动量
            frames = df_safe['Frame'].unique()
            frames.sort()
            
            df_sorted = df_safe.sort_values(by=['VertexID', 'Frame'])
            
            df_sorted['dX'] = df_sorted.groupby('VertexID')['X'].diff().fillna(0)
            df_sorted['dY'] = df_sorted.groupby('VertexID')['Y'].diff().fillna(0)
            df_sorted['dZ'] = df_sorted.groupby('VertexID')['Z'].diff().fillna(0)
            df_sorted['dist'] = np.sqrt(df_sorted['dX']**2 + df_sorted['dY']**2 + df_sorted['dZ']**2)
            
            frame_movement = df_sorted.groupby('Frame')['dist'].sum()
            
            # 3. 寻找静止截断点
            fps = 20
            baseline_frames = int(0.5 * fps)
            if len(frame_movement) < baseline_frames: baseline_frames = len(frame_movement)
            
            baseline_val = frame_movement.iloc[:baseline_frames].mean()
            if baseline_val < 1e-3: baseline_val = 1e-3 
            
            stop_threshold = baseline_val * 0.10 # 阈值 10%
            
            cut_frame = frames[-1]
            consecutive_static = 0
            
            # 从基准后开始扫描
            for f in frames[baseline_frames:]:
                mov = frame_movement.get(f, 0)
                if mov < stop_threshold:
                    consecutive_static += 1
                    # 连续静止超过 15 帧 (约0.75s)
                    if consecutive_static >= 15:
                        cut_frame = f - 15
                        break
                else:
                    consecutive_static = 0
            
            cut_time = df[df['Frame'] == cut_frame]['Time'].iloc[0]
            if cut_time < 0.5: 
                cut_time = max_time
                cut_frame = frames[-1]
                msg = f"未检测到静止，保留全长 {cut_time:.2f}s"
            else:
                msg = f"检测到静止，已裁剪至 {cut_time:.2f}s"
                
            df_trimmed = df[df['Time'] <= cut_time].copy()
            
            # 4. 执行循环
            if loop_count > 1:
                original_chunk = df_trimmed.copy()
                max_frame_idx = df_trimmed['Frame'].max()
                duration = df_trimmed['Time'].max()
                dt = 1.0 / fps 
                
                chunks = [original_chunk]
                
                for i in range(1, loop_count):
                    new_chunk = original_chunk.copy()
                    time_offset = i * (duration + dt) 
                    frame_offset = i * (max_frame_idx + 1)
                    
                    new_chunk['Time'] += time_offset
                    new_chunk['Frame'] += frame_offset
                    chunks.append(new_chunk)
                    
                df_final = pd.concat(chunks, ignore_index=True)
                msg += f" | 已循环 {loop_count} 次"
            else:
                df_final = df_trimmed
            
            cols_to_keep = ["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"]
            df_final = df_final[cols_to_keep]
            
            df_final.to_csv(output_csv, index=False)
            return True, msg
            
        except Exception as e:
            traceback.print_exc()
            return False, f"处理失败: {str(e)}"

    def optimize_trajectory(self, csv_file, safe_dist, max_vel, bound_L, bound_W, bound_H, manual_time_scale=None):
        try:
            df = pd.read_csv(csv_file)
            if df.empty: return False, "Empty", "", {}
            
            all_x = df['X'].values; all_y = df['Y'].values; all_z = df['Z'].values
            min_x, max_x = np.min(all_x), np.max(all_x); min_y, max_y = np.min(all_y), np.max(all_y); min_z, max_z = np.min(all_z), np.max(all_z)
            data_W, data_D, data_H = max_x - min_x, max_y - min_y, max_z - min_z
            
            df['X'] -= (min_x + max_x) / 2.0; df['Y'] -= (min_y + max_y) / 2.0; df['Z'] -= (min_z + max_z) / 2.0
            
            if data_W < 1e-4: data_W = 1e-4
            if data_D < 1e-4: data_D = 1e-4
            if data_H < 1e-4: data_H = 1e-4
            uniform_scale = min(bound_L/data_W, bound_W/data_D, bound_H/data_H) * 0.90
            
            df['X'] *= uniform_scale; df['Y'] *= uniform_scale; df['Z'] *= uniform_scale
            
            df = self.apply_physics_repulsion(df, safe_dist)
            df = self.smooth_trajectory_savgol(df)
            
            df['Z'] += bound_H / 2.0
            
            df = df.sort_values(by=['Object', 'VertexID', 'Frame'])
            dt = df['Time'].diff().mean(); 
            if dt <= 0 or np.isnan(dt): dt = 0.05
            
            try:
                df['dX'] = df['X'].diff(); df['dY'] = df['Y'].diff(); df['dZ'] = df['Z'].diff(); df['ID_diff'] = df['VertexID'].diff()
                valid_moves = df[df['ID_diff'] == 0]
                dist_sq = valid_moves['dX']**2 + valid_moves['dY']**2 + valid_moves['dZ']**2
                max_dist_step = np.sqrt(dist_sq.max()) if not dist_sq.empty else 0
                curr_max_vel = max_dist_step / dt
            except: curr_max_vel = 10.0 
            
            time_scale = 1.0
            if curr_max_vel > max_vel: time_scale = (curr_max_vel / max_vel) * 1.1 
            if manual_time_scale and manual_time_scale > 0: time_scale = manual_time_scale
            df['Time'] *= time_scale
            
            output_file = csv_file 
            df.to_csv(output_file, index=False)
            
            info = {'spatial_scale': uniform_scale, 'time_scale': time_scale, 'orig_min_dist': 0, 'final_max_vel': curr_max_vel / time_scale}
            return True, f"V4.5 优化完成!", output_file, info
            
        except Exception as e:
            traceback.print_exc()
            return False, f"优化失败: {str(e)}", "", {}

    @staticmethod
    def apply_physics_repulsion(df, safe_dist):
        frames = df['Frame'].unique()
        df_sorted = df.sort_values(by=['Frame', 'VertexID'])
        num_frames = len(frames)
        if num_frames == 0: return df
        frame0 = df_sorted[df_sorted['Frame'] == frames[0]]
        num_drones = len(frame0)
        
        if len(df_sorted) != num_frames * num_drones: return df

        coords = df_sorted[['X', 'Y', 'Z']].values.reshape(num_frames, num_drones, 3)
        iterations = 3; stiffness = 0.5 
        for f in range(num_frames):
            pts = coords[f]
            for _ in range(iterations):
                tree = cKDTree(pts); pairs = tree.query_pairs(r=safe_dist)
                if not pairs: break
                for i, j in pairs:
                    p1 = pts[i]; p2 = pts[j]; vec = p1 - p2; dist = np.linalg.norm(vec)
                    if dist < 1e-4: vec = np.random.rand(3) * 0.01; dist = 0.01
                    push_amt = (safe_dist - dist) * 0.5 * stiffness; push_vec = (vec / dist) * push_amt
                    pts[i] += push_vec; pts[j] -= push_vec
            coords[f] = pts
        df_sorted[['X', 'Y', 'Z']] = coords.reshape(-1, 3)
        return df_sorted

    @staticmethod
    def smooth_trajectory_savgol(df):
        df = df.sort_values(by=['Object', 'VertexID', 'Frame'])
        def safe_savgol(x): return savgol_filter(x, 11, 3) if len(x) > 11 else x
        for c in ['X', 'Y', 'Z']: df[c] = df.groupby('VertexID')[c].transform(safe_savgol)
        return df