import fbx
import sys
import os 
import csv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import make_interp_spline 
from scipy.signal import savgol_filter 
from config_manager import cfg
import texture_sampler  
import traceback
import matplotlib.colors as mcolors

class DataExtractor:
    def __init__(self):
        self.all_x = []
        self.all_y = []
        self.all_z =[]
        self.all_c = []
        self.all_mesh_names =[]
        self.all_vertex_ids =[]
        self.fbx_dir = "." 

    @staticmethod
    def boost_night_sky_visibility(colors, min_brightness=0.4, gamma=0.8):
        colors_clip = np.clip(colors, 0.0, 1.0)
        
        hsv_colors = mcolors.rgb_to_hsv(colors_clip)
        
        h = hsv_colors[:, 0]
        s = hsv_colors[:, 1]
        v = hsv_colors[:, 2]
        
        v = np.power(v, gamma)
        
        v = np.clip(v, min_brightness, 1.0)
        
        s = np.clip(s * 1.2, 0.0, 1.0)
        
        hsv_enhanced = np.column_stack((h, s, v))
        rgb_enhanced = mcolors.hsv_to_rgb(hsv_enhanced)
        return rgb_enhanced

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
        
        if np.std(colors) < 0.05:
            print(" 贴图可能丢失，应用'彩虹高度图'")
            z_vals = points[:, 2]
            if len(z_vals) > 0:
                z_min, z_max = z_vals.min(), z_vals.max()
                z_range = z_max - z_min + 0.001
                for i in range(len(colors)):
                    h = (z_vals[i] - z_min) / z_range
                    r = max(0, min(1, 2 * h - 0.5))
                    g = max(0, min(1, 1 - 2 * abs(h - 0.5)))
                    b = max(0, min(1, 1 - 2 * h))
                    colors[i] =[r, g, b]

        colors = self.boost_night_sky_visibility(colors, min_brightness=0.4, gamma=0.7)

        mesh_names = np.array(self.all_mesh_names)
        vertex_ids = np.array(self.all_vertex_ids)

        np.savez("model_data.npz", 
                 points=points, colors=colors, 
                 mesh_names=mesh_names, vertex_ids=vertex_ids)
        
        return True, f"提取完成qwq！原始采样: {len(self.all_x)} 点 "

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

class FormationOptimizer:
    def run(self, axis_mode, target_count, safety_distance):
        try:
            pts, cols, nms, ids = self._load_and_fix_data("model_data.npz", axis_mode)
            if len(pts) == 0: return False, "加载失败", None, None

            HARD_LIMIT = 20000 
            if len(pts) > HARD_LIMIT:
                indices = np.random.choice(len(pts), HARD_LIMIT, replace=False)
                pts = pts[indices]; cols = cols[indices]; nms = nms[indices]; ids = ids[indices]
            
            if not np.isfinite(pts).all(): pts = np.nan_to_num(pts, nan=0.0)
            center = np.mean(pts, axis=0)
            pts_centered = pts - center
            max_range = np.max(np.abs(pts_centered))
            if max_range < 1e-5: max_range = 1.0
            
            scale_factor = 50.0 / max_range 
            pts_norm = pts_centered * scale_factor
            
            if len(pts) > target_count:
                opt_indices = self._adaptive_poisson_disk_sample(pts_norm, cols, target_count)
            else:
                opt_indices = np.arange(len(pts))

            fin_pts = pts_centered[opt_indices]
            fin_cols = cols[opt_indices]
            fin_nms = nms[opt_indices]
            fin_ids = ids[opt_indices]
            
            diff = target_count - len(fin_pts)
            if diff > 0 and len(pts) > 0:
                add_idx = np.random.choice(len(pts), diff, replace=True)
                fin_pts = np.vstack((fin_pts, pts_centered[add_idx]))
                fin_cols = np.vstack((fin_cols, cols[add_idx]))
                fin_nms = np.concatenate((fin_nms, nms[add_idx]))
                fin_ids = np.concatenate((fin_ids, ids[add_idx]))
            
            fin_pts = self._pre_relax(fin_pts, safety_distance * 0.8)
            fin_pts = fin_pts - np.mean(fin_pts, axis=0)

            np.savez("final_formation.npz", 
                     mesh_names=fin_nms, vertex_ids=fin_ids,
                     ref_points=fin_pts, ref_colors=fin_cols)
            
            return True, f"算法就绪! 基于颜色梯度的近似泊松盘生成: {len(fin_pts)} 点", fin_pts, fin_cols
        except Exception as e:
            traceback.print_exc()
            return False, f"优化逻辑发生异常: {str(e)}", None, None

    def _calculate_shannon_entropy(self, colors):
        colors_clip = np.clip(colors, 0.0, 1.0)
        gray = 0.299 * colors_clip[:,0] + 0.587 * colors_clip[:,1] + 0.114 * colors_clip[:,2]
        hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=True)
        p = hist / np.sum(hist)
        p = p[p > 0] 
        entropy = -np.sum(p * np.log2(p))
        return entropy

    def _adaptive_poisson_disk_sample(self, points, colors, target_count):
        N = len(points)
        print("流形表面的高低频颜色纹理差分近似")
        
        entropy = self._calculate_shannon_entropy(colors)
        beta = 1.0 + (entropy / 8.0) * 3.0 
        print(f"测定模型视觉信息熵(Shannon Entropy): {entropy:.2f} -> 动态分发聚焦因子 Beta: {beta:.2f}")

        tree = cKDTree(points)
        dists, idxs = tree.query(points, k=6) 
        
        colors_norm = np.clip(colors, 0.0, 1.0)
        neighbor_colors = colors_norm[idxs] 
        
        color_diffs = np.linalg.norm(neighbor_colors - colors_norm[:, None, :], axis=2) 
        G_vi = np.max(color_diffs, axis=1) 
        
        G_min, G_max = np.min(G_vi), np.max(G_vi)
        if G_max - G_min > 1e-5:
            S_vi = (G_vi - G_min) / (G_max - G_min)
        else:
            S_vi = np.zeros(N)
            
        S_vi = np.power(S_vi, 2)
        
        print("  进行多目标边界下的动态半径推移演算")
        r_low = 0.01
        r_high = 20.0
        best_sel =[]
        best_diff = float('inf')
        
        priority_queue = np.argsort(S_vi)[::-1]
        
        for iter_step in range(15): 
            r_base = (r_low + r_high) / 2.0
            
            r_local = r_base / (1.0 + beta * S_vi)
            
            active = np.ones(N, dtype=bool)
            selected =[]
            
            for i in priority_queue:
                if not active[i]: continue 
                selected.append(i)
                conflict_idxs = tree.query_ball_point(points[i], r_local[i])
                active[conflict_idxs] = False
                
            current_count = len(selected)
            diff = abs(current_count - target_count)
            
            if diff < best_diff:
                best_diff = diff
                best_sel = selected
                
            if current_count > target_count:
                r_low = r_base
            else:
                r_high = r_base
                
            if diff <= int(target_count * 0.01): 
                break
                
        print(f"退火求解结束，最优基准采样点达 {len(best_sel)} 个。")
        best_sel = np.array(best_sel, dtype=int)
        
        final_count = len(best_sel)
        if final_count > target_count:
            best_sel = best_sel[:target_count]
        elif final_count < target_count:
            rem_idx = list(set(range(N)) - set(best_sel))
            diff_lack = target_count - final_count
            if len(rem_idx) >= diff_lack:
                add_idx = np.random.choice(rem_idx, diff_lack, replace=False)
            else:
                add_idx = np.random.choice(range(N), diff_lack, replace=True)
            best_sel = np.concatenate((best_sel, add_idx))
            
        return best_sel

    def _pre_relax(self, points, min_dist):
        if len(points) == 0: return points
        for _ in range(10): 
            tree = cKDTree(points)
            pairs = tree.query_pairs(r=min_dist)
            if not pairs: break
            idx_list = list(pairs)
            p1_idx =[i for i, j in idx_list]; p2_idx =[j for i, j in idx_list]
            p1 = points[p1_idx]; p2 = points[p2_idx]
            vec = p1 - p2
            dist = np.linalg.norm(vec, axis=1)
            dist[dist < 1e-5] = 1e-5
            push = (vec / dist[:, None]) * (min_dist - dist[:, None]) * 0.5
            np.add.at(points, p1_idx, push)
            np.add.at(points, p2_idx, -push)
        return points

    def _load_and_fix_data(self, file_path, mode):
        try:
            data = np.load(file_path)
            points = data['points']; colors = data['colors']; names = data['mesh_names']; ids = data['vertex_ids']
            if mode == 1: points = points[:,[0, 2, 1]]
            elif mode == 2: points = points[:, [2, 1, 0]]
            elif mode == 3: points = points[:, [1, 0, 2]]
            elif mode == 4: points[:, 1], points[:, 2] = points[:, 2].copy(), -points[:, 1].copy()
            return points, colors, names, ids
        except: return [], [],[],[]

class AnimationExporter:
    def __init__(self):
        self.TARGET_MAP = {}
        self.COLOR_MAP = {}
        self.SKINNING_DATA = {}

    def get_animations(self, fbx_file):
        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "Scene")
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(fbx_file, -1, manager.GetIOSettings()): return[]
        importer.Import(scene); importer.Destroy(); criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        num = scene.GetSrcObjectCount(criteria); anims =[]
        for i in range(num):
            s = scene.GetSrcObject(criteria, i); span = s.GetLocalTimeSpan(); dur = span.GetStop().GetSecondDouble() - span.GetStart().GetSecondDouble()
            anims.append({"name": s.GetName(), "duration": dur, "index": i})
        manager.Destroy(); return anims

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
            mesh = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId), i)
            node = mesh.GetNode()
            if not node or node.GetName() not in self.TARGET_MAP: continue
            
            m_name = node.GetName()
            target_ids = self.TARGET_MAP[m_name]
            skin_deformer = None
            
            for j in range(mesh.GetDeformerCount()):
                if mesh.GetDeformer(j).GetClassId() == fbx.FbxSkin.ClassId: 
                    skin_deformer = mesh.GetDeformer(j)
                    break
                    
            if not skin_deformer: continue 
            
            for c_idx in range(skin_deformer.GetClusterCount()):
                cluster = skin_deformer.GetCluster(c_idx)
                bone = cluster.GetLink()
                if not bone: continue
                
                # 获取骨骼绑定逆矩阵
                lMatrix = fbx.FbxAMatrix()
                cluster.GetTransformLinkMatrix(lMatrix)
                bind_inv = lMatrix.Inverse()
                
                # 【新增核心逻辑 1】：获取网格(Mesh)在绑定骨骼那一刻的世界矩阵
                transform_matrix = fbx.FbxAMatrix()
                cluster.GetTransformMatrix(transform_matrix)
                
                ind = cluster.GetControlPointIndices()
                wht = cluster.GetControlPointWeights()
                
                for k in range(cluster.GetControlPointIndicesCount()):
                    v_idx = ind[k]
                    if v_idx in target_ids:
                        # 【修改】：将 transform_matrix 一起存入字典
                        self.SKINNING_DATA.setdefault((m_name, v_idx), []).append((bone, bind_inv, transform_matrix, wht[k]))

    def _process_node_raw(self, node, start, frames, fps, scale, mode, writer):
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            if node.GetName() in self.TARGET_MAP:
                self._extract_data_raw(node, start, frames, fps, scale, mode, writer)
        for i in range(node.GetChildCount()):
            self._process_node_raw(node.GetChild(i), start, frames, fps, scale, mode, writer)

    def _extract_data_raw(self, node, start, frames, fps, scale, mode, writer):
        mesh = node.GetMesh()
        m_name = node.GetName()
        target_ids = self.TARGET_MAP[m_name]
        l_verts = mesh.GetControlPoints()
        t = fbx.FbxTime()
        
        for f in range(frames + 1):
            curr_time = f / fps
            t.SetSecondDouble(start + curr_time)
            
            g_trans = node.EvaluateGlobalTransform(t)
            
            for v_idx in target_ids:
                final = fbx.FbxVector4(0, 0, 0, 0)
                
                if (m_name, v_idx) in self.SKINNING_DATA:
                    skinning_list = self.SKINNING_DATA[(m_name, v_idx)]
                    
                    # 【新增核心逻辑 2】：计算总权重，防止某些野生 FBX 模型的权重和不等于 1.0 导致模型被撕裂
                    total_weight = sum([w for _, _, _, w in skinning_list])
                    
                    for bone, binv, tmat, w in skinning_list:
                        # 权重归一化保护
                        normalized_weight = w / total_weight if total_weight > 0 else 0.0
                        
                        # 【核心公式修正】：Bone动画矩阵 * Bone绑定逆矩阵 * Mesh绑定矩阵
                        cluster_matrix = bone.EvaluateGlobalTransform(t) * binv * tmat
                        
                        final += cluster_matrix.MultT(l_verts[v_idx]) * normalized_weight
                else: 
                    # 如果没有绑定骨骼，使用节点自身的全局矩阵
                    final = g_trans.MultT(l_verts[v_idx])
                    
                x, y, z = final[0]*scale, final[1]*scale, final[2]*scale
                
                # 坐标系矫正逻辑保持不变
                if mode == 1: x, y, z = x, z, y
                elif mode == 2: x, y, z = z, y, x
                elif mode == 3: x, y, z = y, x, z
                elif mode == 4: y, z = z, -y
                
                rgb = self.COLOR_MAP.get((m_name, v_idx), [1, 1, 1])
                writer.writerow([f, f"{curr_time:.3f}", m_name, v_idx, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)])

class TrajectoryOptimizer:
    def smart_trim_and_loop(self, input_csv, output_csv, loop_count):
        try:
            df = pd.read_csv(input_csv)
            if df.empty: return False, "数据为空"
            
            max_time = df['Time'].max()
            safe_end = max_time - 0.5
            if safe_end < 0.5: safe_end = max_time
            
            df_safe = df[df['Time'] <= safe_end]
            frames = df_safe['Frame'].unique()
            frames.sort()
            
            df_sorted = df_safe.sort_values(by=['VertexID', 'Frame'])
            df_sorted['dX'] = df_sorted.groupby('VertexID')['X'].diff().fillna(0)
            df_sorted['dY'] = df_sorted.groupby('VertexID')['Y'].diff().fillna(0)
            df_sorted['dZ'] = df_sorted.groupby('VertexID')['Z'].diff().fillna(0)
            df_sorted['dist'] = np.sqrt(df_sorted['dX']**2 + df_sorted['dY']**2 + df_sorted['dZ']**2)
            
            frame_movement = df_sorted.groupby('Frame')['dist'].sum()
            
            fps = cfg.default_fps
            baseline_frames = int(0.5 * fps)
            if len(frame_movement) < baseline_frames: baseline_frames = len(frame_movement)
            
            baseline_val = frame_movement.iloc[:baseline_frames].mean()
            if baseline_val < 1e-3: baseline_val = 1e-3 
            
            stop_threshold = baseline_val * 0.10 
            
            cut_frame = frames[-1]
            consecutive_static = 0
            
            for f in frames[baseline_frames:]:
                mov = frame_movement.get(f, 0)
                if mov < stop_threshold:
                    consecutive_static += 1
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
            
            if loop_count > 1:
                original_chunk = df_trimmed.copy()
                max_frame_idx = df_trimmed['Frame'].max()
                duration = df_trimmed['Time'].max()
                dt = 1.0 / fps 
                chunks =[original_chunk]
                
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
            
            cols_to_keep =["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"]
            df_final = df_final[cols_to_keep]
            
            df_final.to_csv(output_csv, index=False)
            return True, msg
            
        except Exception as e:
            traceback.print_exc()
            return False, f"处理失败: {str(e)}"

    def optimize_trajectory(self, csv_file, safe_dist, max_vel, bound_L, bound_W, bound_H, manual_time_scale=None):
        try:
            df = pd.read_csv(csv_file)
            if df.empty: return False, "数据为空", "", {}
            
            # --- 1. 空间归一化与中心化 ---
            all_x = df['X'].values; all_y = df['Y'].values; all_z = df['Z'].values
            min_x, max_x = np.min(all_x), np.max(all_x)
            min_y, max_y = np.min(all_y), np.max(all_y)
            min_z, max_z = np.min(all_z), np.max(all_z)
            data_W, data_D, data_H = max_x - min_x, max_y - min_y, max_z - min_z
            
            # 将原始阵型居中，方便后续统一拉伸
            df['X'] -= (min_x + max_x) / 2.0
            df['Y'] -= (min_y + max_y) / 2.0
            df['Z'] -= (min_z + max_z) / 2.0
            
            # ================= 致命空间陷阱修复开始 =================
            first_frame_df = df[df['Frame'] == df['Frame'].min()]
            temp_pts = first_frame_df[['X', 'Y', 'Z']].values
            
            if len(temp_pts) > 1:
                tree = cKDTree(temp_pts)
                dists, _ = tree.query(temp_pts, k=2)
                # 【修复1】：废弃平均值，寻找阵型中最拥挤（距离最短）的两架飞机作为拉伸基准
                current_min_dist = np.min(dists[:, 1]) 
                current_avg_dist = np.mean(dists[:, 1]) # 保留平均值用于备用视觉参考
            else:
                current_min_dist = safe_dist
                current_avg_dist = safe_dist
                
            if current_min_dist < 1e-4: current_min_dist = 1e-4
            
            # 底线原则 A (保命)：必须把最挤的两架飞机拉开到 safe_dist (留 5% 物理冗余)
            safe_density_scale = (safe_dist * 1.05) / current_min_dist
            
            # 底线原则 B (合规)：整个阵型不能撞到用户设定的长宽高边界虚拟墙
            box_limit_scale = min(bound_L/max(data_W, 1e-4), bound_W/max(data_D, 1e-4), bound_H/max(data_H, 1e-4)) * 0.90
            
            # 【修复2】：冲突仲裁机制
            conflict_msg = ""
            if safe_density_scale > box_limit_scale:
                # 当安全与边界发生冲突时，绝对向安全妥协！
                uniform_scale = safe_density_scale
                conflict_msg = f"\n   ⚠️ [空间妥协] 场地 {bound_L}x{bound_W} 偏小！为强制保障 {safe_dist}m 的防撞底线，阵型已被迫突破您设置的边界参数。"
            else:
                # 场地足够大：在绝对安全的基础上，兼顾原有的视觉张力
                visual_scale = (safe_dist * 1.2) / max(current_avg_dist, 1e-4)
                # 确保它既满足最小安全底线，又不超过场地盒子，且保持一定视觉美感
                uniform_scale = max(safe_density_scale, min(visual_scale, box_limit_scale))
                
            # 执行最终的空间物理拉伸
            df['X'] *= uniform_scale
            df['Y'] *= uniform_scale
            df['Z'] *= uniform_scale
            # ================= 空间陷阱修复结束 =================

            # 调用已升级的动力学势场避障 (APF)
            df = self.apply_physics_repulsion(df, safe_dist)
            
            # 重新排序，准备进行时间维度的物理运算
            df = df.sort_values(by=['Object', 'VertexID', 'Frame'])
            
            orig_dt = 1.0 / cfg.default_fps
            df['dX'] = df.groupby('VertexID')['X'].diff().fillna(0.0)
            df['dY'] = df.groupby('VertexID')['Y'].diff().fillna(0.0)
            df['dZ'] = df.groupby('VertexID')['Z'].diff().fillna(0.0)
            df['dT'] = df.groupby('VertexID')['Time'].diff().fillna(orig_dt)
            df['dT'] = df['dT'].replace(0, orig_dt)
            
            df['Vel_X'] = df['dX'] / df['dT']
            df['Vel_Y'] = df['dY'] / df['dT']
            df['Vel_Z'] = df['dZ'] / df['dT']
            df['Vel'] = np.sqrt(df['Vel_X']**2 + df['Vel_Y']**2 + df['Vel_Z']**2)
            
            df['dVel_X'] = df.groupby('VertexID')['Vel_X'].diff().fillna(0.0)
            df['dVel_Y'] = df.groupby('VertexID')['Vel_Y'].diff().fillna(0.0)
            df['dVel_Z'] = df.groupby('VertexID')['Vel_Z'].diff().fillna(0.0)
            df['Acc'] = np.sqrt(df['dVel_X']**2 + df['dVel_Y']**2 + df['dVel_Z']**2) / df['dT']
            
            curr_max_vel = df['Vel'].max()
            curr_max_acc = df['Acc'].max()
            
            max_acc = cfg.max_accel 
            
            # ================= 时间拉伸降速自保机制 =================
            s_vel = curr_max_vel / max_vel if curr_max_vel > max_vel else 1.0
            s_acc = np.sqrt(curr_max_acc / max_acc) if curr_max_acc > max_acc else 1.0
            s = max(1.0, s_vel, s_acc)
            if manual_time_scale and manual_time_scale > 1.0: 
                s = max(s, manual_time_scale)
            
            # 将物理时间拉长，压制速度和加速度
            df['Time'] = df['Time'] * s
            # ========================================================
            
            # 高频 B 样条重采样平滑曲线
            df = self.smooth_trajectory_b_spline(df, dt_sample=1.0/cfg.high_density_fps)
            
            # 整体拔高，放置在边界的 H/2 高度上（保证最低点不砸地）
            df['Z'] += bound_H / 2.0
            
            final_max_vel = curr_max_vel / s
            final_max_acc = curr_max_acc / (s**2)
            
            df.to_csv(csv_file, index=False)
            info = {'spatial_scale': uniform_scale, 'time_scale': s, 'orig_max_vel': curr_max_vel, 'final_max_vel': final_max_vel}
            
            # 组装 UI 报告
            msg = f"✅ 重构与优化完成\n"
            msg += f"  📐 空间缩放: 放大了 {uniform_scale:.2f} 倍{conflict_msg}\n"
            msg += f"  🏎️ 原始极限速: {curr_max_vel:.2f} m/s | 原始极限过载: {curr_max_acc:.2f} m/s²\n"
            if s > 1.0:
                msg += f"  ⏳ 降速处理: 全局时间拉伸慢放 {s:.2f} 倍\n"
            msg += f"  🛡️ 物理下发指标 -> 速度: {final_max_vel:.2f} m/s | 加速度: {final_max_acc:.2f} m/s²"
            
            return True, msg, csv_file, info
            
        except Exception as e:
            traceback.print_exc()
            return False, f"优化失败: {str(e)}", "", {}
    
    @staticmethod
    def apply_physics_repulsion(df, safe_dist):
        """
        动力学人工势场避障算法 (Kinematic APF with Damping & Curl Field)
        """
        df_sorted = df.sort_values(by=['Frame', 'Object', 'VertexID'])
        frames = df_sorted['Frame'].unique()
        num_frames = len(frames)
        if num_frames == 0: return df

        frame0 = df_sorted[df_sorted['Frame'] == frames[0]]
        num_drones = len(frame0)
        
        if len(df_sorted) != num_frames * num_drones: 
            return df

        coords_target = df_sorted[['X', 'Y', 'Z']].values.reshape(num_frames, num_drones, 3)
        coords_actual = np.zeros_like(coords_target)
        
        # ================= 核心物理系统参数配置 =================
        dt = 1.0 / cfg.default_fps
        k_attract = 12.0            # 目标引力系数
        k_repel = 35.0              # 径向斥力系数 (防撞)
        k_curl = 0               # [新增] 旋度场系数 (控制绕行侧滑的力度)
        damping = 0.85              # 阻尼系数
        # ========================================================

        coords_actual[0] = coords_target[0].copy()
        velocities = np.zeros((num_drones, 3))

        for f in range(1, num_frames):
            curr_pos = coords_actual[f-1].copy()
            target_pos = coords_target[f]

            F_attract = k_attract * (target_pos - curr_pos)
            F_repel = np.zeros((num_drones, 3))
            
            tree = cKDTree(curr_pos)
            defense_radius = safe_dist * 1.25 
            pairs = tree.query_pairs(r=defense_radius)

            z_axis = np.array([0.0, 0.0, 1.0]) # 垂直参考轴

            for i, j in pairs:
                vec = curr_pos[i] - curr_pos[j]
                dist = np.linalg.norm(vec)
                
                if dist < 1e-4:
                    vec = np.random.rand(3) * 0.01 - 0.005
                    dist = np.linalg.norm(vec)

                # 1. 基础径向斥力 (互相推开)
                push_force = k_repel * (defense_radius - dist)
                repel_vec = (vec / dist) * push_force

                # 2. [新增] 旋度场切向力 (侧身绕行)
                # 使用叉乘产生水平侧向引导力
                curl_vec = np.cross(vec / dist, z_axis)
                
                # 极端情况保护：如果两架飞机恰好在垂直 Z 轴上绝对重合
                if np.linalg.norm(curl_vec) < 1e-3:
                    curl_vec = np.array([1.0, 0.0, 0.0]) # 强制给个X轴偏置
                else:
                    curl_vec = curl_vec / np.linalg.norm(curl_vec)

                # 旋度力的大小与斥力成正比，距离越近，急转弯的力越大
                curl_force = curl_vec * push_force * k_curl

                # 3. 最终合力 = 推开 + 侧滑
                total_force = repel_vec + curl_force

                # 作用力与反作用力：对方受到相反的旋度力，形成太极双螺旋
                F_repel[i] += total_force
                F_repel[j] -= total_force

            acceleration = F_attract + F_repel
            velocities = velocities + acceleration * dt
            velocities = velocities * damping
            coords_actual[f] = curr_pos + velocities * dt

        df_sorted[['X', 'Y', 'Z']] = coords_actual.reshape(-1, 3)
        return df_sorted

    """ @staticmethod
    def apply_physics_repulsion(df, safe_dist):
        df_sorted = df.sort_values(by=['Frame', 'Object', 'VertexID'])
        frames = df_sorted['Frame'].unique()
        num_frames = len(frames)
        if num_frames == 0: return df

        frame0 = df_sorted[df_sorted['Frame'] == frames[0]]
        num_drones = len(frame0)
        
        # 帧数和无人机数量匹配
        if len(df_sorted) != num_frames * num_drones: 
            return df

        # 提取目标动画轨迹
        coords_target = df_sorted[['X', 'Y', 'Z']].values.reshape(num_frames, num_drones, 3)
        
        # 创建飞行轨迹容器
        coords_actual = np.zeros_like(coords_target)
        
        # 参数配置
        dt = 1.0 / cfg.default_fps  # 物理仿真步长
        k_attract = 12.0            # 目标引力系数
        k_repel = 35.0              # 避障斥力系数
        damping = 0.85              # 空气阻尼系数 (0~1)

        # 初始状态
        coords_actual[0] = coords_target[0].copy()
        velocities = np.zeros((num_drones, 3))

        for f in range(1, num_frames):
            curr_pos = coords_actual[f-1].copy() # 当前真实的物理坐标
            target_pos = coords_target[f]        # 动画驱动的虚拟目标坐标

            # 1. 计算引力场 (Attraction Force)
            F_attract = k_attract * (target_pos - curr_pos)

            # 2. 计算斥力场 (Repulsion Force)
            F_repel = np.zeros((num_drones, 3))
            tree = cKDTree(curr_pos)
            
            defense_radius = safe_dist * 1.25 
            pairs = tree.query_pairs(r=defense_radius)

            for i, j in pairs:
                vec = curr_pos[i] - curr_pos[j]
                dist = np.linalg.norm(vec)
                
                if dist < 1e-4:
                    vec = np.random.rand(3) * 0.01 - 0.005
                    dist = np.linalg.norm(vec)

                push_force = k_repel * (defense_radius - dist)
                force_vec = (vec / dist) * push_force

                F_repel[i] += force_vec
                F_repel[j] -= force_vec

            # 3. 运动学解算 
            # 假设无人机质量 m=1.0，则 F = ma -> a = F
            acceleration = F_attract + F_repel

            # 4. 半隐式欧拉积分
            velocities = velocities + acceleration * dt
            
            velocities = velocities * damping

            # 更新下一帧的物理位置
            coords_actual[f] = curr_pos + velocities * dt

        # 实际坐标写回 DataFrame
        df_sorted[['X', 'Y', 'Z']] = coords_actual.reshape(-1, 3)
        return df_sorted """


    @staticmethod
    def smooth_trajectory_b_spline(df, dt_sample=0.02):
        df = df.sort_values(by=['Object', 'VertexID', 'Frame'])
        frames = df['Frame'].unique()
        frames.sort()
        
        step_size = 2 
        key_frames = frames[::step_size]
        if frames[-1] not in key_frames:
            key_frames = np.append(key_frames, frames[-1])
            
        new_rows =[]
        for vid, group in df.groupby('VertexID'):
            times = group['Time'].values
            kf_mask = np.isin(group['Frame'], key_frames)
            kf_times = times[kf_mask]
            
            obj_name = group['Object'].iloc[0]
            r, g, b = group['R'].iloc[0], group['G'].iloc[0], group['B'].iloc[0]
            
            if len(kf_times) >= 4:
                bc = ([(1, 0.0), (2, 0.0)],[(1, 0.0), (2, 0.0)])
                try:
                    sp_x = make_interp_spline(kf_times, group.loc[kf_mask, 'X'].values, k=3, bc_type=bc)
                    sp_y = make_interp_spline(kf_times, group.loc[kf_mask, 'Y'].values, k=3, bc_type=bc)
                    sp_z = make_interp_spline(kf_times, group.loc[kf_mask, 'Z'].values, k=3, bc_type=bc)
                    
                    # 【绝对等距约束】强制按 dt_sample (0.02s) 在拉长后的时间轴上采样
                    dense_times = np.arange(times[0], times[-1] + 1e-5, dt_sample)
                    x_dense = sp_x(dense_times)
                    y_dense = sp_y(dense_times)
                    z_dense = sp_z(dense_times)
                    
                    for idx, t_val in enumerate(dense_times):
                        new_rows.append([idx, t_val, obj_name, vid, x_dense[idx], y_dense[idx], z_dense[idx], r, g, b])
                    continue
                except Exception:
                    pass
            
            # 退化处理
            for _, row in group.iterrows():
                new_rows.append([row['Frame'], row['Time'], row['Object'], row['VertexID'], row['X'], row['Y'], row['Z'], row['R'], row['G'], row['B']])
                
        new_df = pd.DataFrame(new_rows, columns=["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
        return new_df