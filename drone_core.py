import fbx
import sys
import os 
import csv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
# 引入 B 样条插值库，用于实现闭边界 B 样条重构
from scipy.interpolate import make_interp_spline 
from scipy.signal import savgol_filter 
from config_manager import cfg
import texture_sampler  
import traceback
import matplotlib.colors as mcolors

# ==========================================
# 模块 1: 数据提取 (DataExtractor)
# ==========================================
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
        """
        针对无人机夜空显示的色彩增强算法
        将过暗的像素提亮，防止无人机在夜空中“隐形”
        """
        colors_clip = np.clip(colors, 0.0, 1.0)
        
        # 1. 转换为 HSV (色相、饱和度、明度) 空间
        hsv_colors = mcolors.rgb_to_hsv(colors_clip)
        
        h = hsv_colors[:, 0]
        s = hsv_colors[:, 1]
        v = hsv_colors[:, 2]
        
        # 2. 伽马校正：平滑地提亮全局暗部
        v = np.power(v, gamma)
        
        # 3. 亮度钳制 (Clamp)：强制所有颜色不得低于 min_brightness
        # (默认设定为0.4，即RGB值最低也会被强制拉到约102/255的亮度)
        v = np.clip(v, min_brightness, 1.0)
        
        # 4. 饱和度轻微补偿（避免提亮后颜色发白/发灰）
        s = np.clip(s * 1.2, 0.0, 1.0)
        
        # 重组并转回 RGB
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
        
        # 【增强版颜色保底】
        if np.std(colors) < 0.05:
            print("   [提示] 检测到模型颜色单一（贴图可能丢失），强制应用'彩虹高度图'保底。")
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
# 模块 2: 编队优化 (高阶版：特征感知与自适应泊松盘混合采样)
# 落实中期报告 6.2.1: 引入 Shannon Entropy 动态调节 Beta
# ==========================================
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
        """计算模型整体色彩分布的香农信息熵"""
        colors_clip = np.clip(colors, 0.0, 1.0)
        # 转为灰度值
        gray = 0.299 * colors_clip[:,0] + 0.587 * colors_clip[:,1] + 0.114 * colors_clip[:,2]
        # 统计 256 级灰度直方图
        hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=True)
        # 计算概率分布
        p = hist / np.sum(hist)
        p = p[p > 0] # 排除 0 概率防止 log2 报错
        # 香农信息熵公式
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
            if df.empty: return False, "Empty", "", {}
            
            # --- 空间归一化与智能密度缩放 ---
            all_x = df['X'].values; all_y = df['Y'].values; all_z = df['Z'].values
            min_x, max_x = np.min(all_x), np.max(all_x); min_y, max_y = np.min(all_y), np.max(all_y); min_z, max_z = np.min(all_z), np.max(all_z)
            data_W, data_D, data_H = max_x - min_x, max_y - min_y, max_z - min_z
            df['X'] -= (min_x + max_x) / 2.0; df['Y'] -= (min_y + max_y) / 2.0; df['Z'] -= (min_z + max_z) / 2.0
            
            first_frame_df = df[df['Frame'] == df['Frame'].min()]
            temp_pts = first_frame_df[['X', 'Y', 'Z']].values
            if len(temp_pts) > 1:
                tree = cKDTree(temp_pts); dists, _ = tree.query(temp_pts, k=2)
                current_avg_dist = np.mean(dists[:, 1])
            else:
                current_avg_dist = safe_dist
                
            if current_avg_dist < 1e-4: current_avg_dist = 1e-4
            density_scale = (safe_dist * 1.2) / current_avg_dist
            box_limit_scale = min(bound_L/max(data_W, 1e-4), bound_W/max(data_D, 1e-4), bound_H/max(data_H, 1e-4)) * 0.90
            uniform_scale = min(density_scale, box_limit_scale)
            df['X'] *= uniform_scale; df['Y'] *= uniform_scale; df['Z'] *= uniform_scale
            
            df = self.apply_physics_repulsion(df, safe_dist)
            df = df.sort_values(by=['Object', 'VertexID', 'Frame'])
            
            # ==========================================================
            # 【严谨时序 第一段】：基于原始频率，测算极限运动学参数
            # ==========================================================
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
            
            # 测算加速度
            df['dVel_X'] = df.groupby('VertexID')['Vel_X'].diff().fillna(0.0)
            df['dVel_Y'] = df.groupby('VertexID')['Vel_Y'].diff().fillna(0.0)
            df['dVel_Z'] = df.groupby('VertexID')['Vel_Z'].diff().fillna(0.0)
            df['Acc'] = np.sqrt(df['dVel_X']**2 + df['dVel_Y']**2 + df['dVel_Z']**2) / df['dT']
            
            curr_max_vel = df['Vel'].max()
            curr_max_acc = df['Acc'].max()
            
            # 引入全局配置的极限加速度
            max_acc = cfg.max_accel 
            
            # 计算拉伸系数 S
            s_vel = curr_max_vel / max_vel if curr_max_vel > max_vel else 1.0
            s_acc = np.sqrt(curr_max_acc / max_acc) if curr_max_acc > max_acc else 1.0
            s = max(1.0, s_vel, s_acc)
            if manual_time_scale and manual_time_scale > 1.0: 
                s = max(s, manual_time_scale)
            
            # ==========================================================
            # 【严谨时序 第二段】：先拉伸时间轴，再高频重构！
            # ==========================================================
            df['Time'] = df['Time'] * s
            
            # 此时传入的 df，时间轴已经被拉伸完毕。
            # 函数内部将在拉长后的时间轴上，严格按 dt=0.02s(50Hz) 切片。
            df = self.smooth_trajectory_b_spline(df, dt_sample=1.0/cfg.high_density_fps)
            
            df['Z'] += bound_H / 2.0
            
            final_max_vel = curr_max_vel / s
            final_max_acc = curr_max_acc / (s**2)
            
            df.to_csv(csv_file, index=False)
            info = {'spatial_scale': uniform_scale, 'time_scale': s, 'orig_max_vel': curr_max_vel, 'final_max_vel': final_max_vel}
            
            msg = f"重构完成\n"
            msg += f" 原始极限速: {curr_max_vel:.2f} m/s | 原始极限加速度: {curr_max_acc:.2f} m/s²\n"
            if s > 1.0:
                msg += f"  全局时间拉伸 {s:.2f} 倍\n"
            msg += f"  安全验证速度: {final_max_vel:.2f} m/s | 安全加速度: {final_max_acc:.2f} m/s²"
            
            return True, msg, csv_file, info
            
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
    def smooth_trajectory_b_spline(df, dt_sample=0.02):
        """重新以50Hz生成解析后的B样条平滑轨迹"""
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