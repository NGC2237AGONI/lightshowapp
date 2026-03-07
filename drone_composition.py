import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from drone_core import TrajectoryOptimizer
import traceback

class CompositionManager:
    def __init__(self):
        # 存储格式增加 'position'
        self.playlist = []
        
    def add_file(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_cols = {'Frame', 'Time', 'VertexID', 'X', 'Y', 'Z'}
            if not required_cols.issubset(df.columns):
                return False, "CSV 格式不正确，缺少必要列"
            
            item = {
                'file': csv_path,
                'transition_dur': 5.0,
                'rotation': [0, 0, 0],
                'position': [0, 0, 0], # 新增：默认不位移
                'data': df
            }
            self.playlist.append(item)
            return True, f"已添加: {csv_path}"
        except Exception as e:
            return False, f"读取失败: {str(e)}"
            
    def remove_file(self, index):
        if 0 <= index < len(self.playlist):
            self.playlist.pop(index)
            
    def clear(self):
        self.playlist = []

    def set_transition_duration(self, index, duration):
        if 0 <= index < len(self.playlist):
            self.playlist[index]['transition_dur'] = max(1.0, duration)

    def set_rotation(self, index, x, y, z):
        if 0 <= index < len(self.playlist):
            self.playlist[index]['rotation'] = [x, y, z]

    # 新增：设置位移
    def set_position(self, index, x, y, z):
        if 0 <= index < len(self.playlist):
            self.playlist[index]['position'] = [x, y, z]

    # 【核心逻辑】：应用原地旋转 + 全局位移
    def _apply_transform(self, df, rotation, position):
        """
        1. 计算质心并归零 (Centering)
        2. 旋转 (Rotate)
        3. 复原质心 (Restore) -> 实现原地自转
        4. 应用位移 (Translate) -> 实现位置偏移
        """
        if not rotation and not position:
            return # 无操作
            
        pts = df[['X', 'Y', 'Z']].values
        
        # 1. 计算质心
        centroid = np.mean(pts, axis=0)
        
        # 2. 归零
        pts_centered = pts - centroid
        
        # 3. 旋转 (绕原点，即绕自身的几何中心)
        if any(r != 0 for r in rotation):
            rads = np.radians(rotation)
            rx, ry, rz = rads[0], rads[1], rads[2]
            
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            
            # Z * Y * X order
            R = Rz @ Ry @ Rx
            pts_centered = pts_centered @ R.T
            
        # 4. 复原质心 + 5. 应用用户位移
        final_pts = pts_centered + centroid + np.array(position)
        
        df['X'] = final_pts[:, 0]
        df['Y'] = final_pts[:, 1]
        df['Z'] = final_pts[:, 2]

    # 【新增】：强制边界检查
    def _enforce_boundaries(self, df, bounds):
        """
        检查这一段动画是否飞出了 L/W/H 的范围。
        如果有，整体平移回来，而不是缩放（保持编队形状）。
        """
        L, W, H = bounds
        # X: [-L/2, L/2], Y: [-W/2, W/2], Z: [0, H]
        min_x, max_x = df['X'].min(), df['X'].max()
        min_y, max_y = df['Y'].min(), df['Y'].max()
        min_z, max_z = df['Z'].min(), df['Z'].max()
        
        shift_x = 0
        shift_y = 0
        shift_z = 0
        
        # X轴检查
        if max_x > L/2: shift_x = (L/2) - max_x # 往左推
        elif min_x < -L/2: shift_x = (-L/2) - min_x # 往右推
        
        # Y轴检查
        if max_y > W/2: shift_y = (W/2) - max_y
        elif min_y < -W/2: shift_y = (-W/2) - min_y
        
        # Z轴检查 (高度)
        if max_z > H: shift_z = H - max_z # 往下压
        elif min_z < 0: shift_z = 0 - min_z # 往上提
        
        if shift_x != 0 or shift_y != 0 or shift_z != 0:
            print(f"   [边界修正] 修正偏移: ({shift_x:.2f}, {shift_y:.2f}, {shift_z:.2f})")
            df['X'] += shift_x
            df['Y'] += shift_y
            df['Z'] += shift_z

    def merge_shows(self, output_path, safe_dist, bounds):
        if len(self.playlist) < 1:
            return False, "请至少添加一个文件"
            
        full_df = pd.DataFrame()
        current_time_offset = 0.0
        fps = 20 
        dt = 1.0 / fps
        
        try:
            # === 处理第一个文件 ===
            first_item = self.playlist[0]
            df_curr = first_item['data'].copy()
            
            # 应用变换
            self._apply_transform(df_curr, first_item.get('rotation', [0,0,0]), first_item.get('position', [0,0,0]))
            # 强制边界检查
            self._enforce_boundaries(df_curr, bounds)
            
            df_curr = df_curr.sort_values(by=['Frame', 'VertexID'])
            df_curr['Time'] = df_curr['Time'] - df_curr['Time'].min()
            last_time = df_curr['Time'].max()
            last_frame = df_curr['Frame'].max()
            
            full_df = df_curr
            current_time_offset = last_time
            current_frame_offset = last_frame
            
            last_frame_data = df_curr[df_curr['Frame'] == last_frame].sort_values('VertexID')
            prev_end_pos = last_frame_data[['X', 'Y', 'Z']].values
            prev_ids = last_frame_data['VertexID'].values
            
            # === 循环处理后续文件 ===
            for i in range(1, len(self.playlist)):
                item = self.playlist[i]
                df_next = item['data'].copy()
                trans_dur = self.playlist[i-1]['transition_dur']
                
                # 应用变换 (变换后再进行边界检查和过渡计算)
                self._apply_transform(df_next, item.get('rotation', [0,0,0]), item.get('position', [0,0,0]))
                self._enforce_boundaries(df_next, bounds)
                
                df_next = df_next.sort_values(by=['Frame', 'VertexID'])
                first_frame_next = df_next['Frame'].min()
                start_frame_data = df_next[df_next['Frame'] == first_frame_next].sort_values('VertexID')
                next_start_pos = start_frame_data[['X', 'Y', 'Z']].values
                next_ids = start_frame_data['VertexID'].values
                
                if len(prev_end_pos) != len(next_start_pos):
                    return False, f"文件 {i} 和文件 {i+1} 的无人机数量不一致"
                
                # 匈牙利算法匹配
                dist_matrix = cdist(prev_end_pos, next_start_pos)
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                
                target_to_source_map = {}
                for r, c in zip(row_ind, col_ind):
                    source_id = prev_ids[r]
                    target_id = next_ids[c]
                    target_to_source_map[target_id] = source_id
                
                df_next['VertexID'] = df_next['VertexID'].map(target_to_source_map)
                sorted_next_start_pos = next_start_pos[col_ind]
                
                # 生成过渡
                trans_frames = int(trans_dur * fps)
                trans_df = self._generate_transition_data(
                    prev_end_pos, 
                    sorted_next_start_pos, 
                    trans_frames, 
                    dt, 
                    current_frame_offset, 
                    current_time_offset,
                    prev_ids, 
                    safe_dist
                )
                
                current_time_offset += (trans_dur + dt) 
                current_frame_offset += (trans_frames + 1)
                
                df_next['Time'] = df_next['Time'] - df_next['Time'].min() + current_time_offset
                df_next['Frame'] = df_next['Frame'] - df_next['Frame'].min() + current_frame_offset
                
                full_df = pd.concat([full_df, trans_df, df_next], ignore_index=True)
                
                last_time = df_next['Time'].max()
                last_frame = df_next['Frame'].max()
                current_time_offset = last_time + dt
                current_frame_offset = last_frame + 1
                
                last_frame_data = df_next[df_next['Frame'] == last_frame].sort_values('VertexID')
                prev_end_pos = last_frame_data[['X', 'Y', 'Z']].values
                prev_ids = last_frame_data['VertexID'].values
                
            cols = ["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"]
            full_df = full_df[cols]
            full_df.to_csv(output_path, index=False)
            
            return True, f"合成成功！总时长 {full_df['Time'].max():.2f}秒"

        except Exception as e:
            traceback.print_exc()
            return False, f"合成出错: {str(e)}"

    def _generate_transition_data(self, start_pos, end_pos, frames, dt, start_frame, start_time, ids, safe_dist):
        num_drones = len(start_pos)
        data_list = []
        
        for f in range(1, frames + 1): 
            t = f / float(frames + 1) 
            factor = 0.5 - 0.5 * np.cos(np.pi * t)
            curr_pos = start_pos + (end_pos - start_pos) * factor
            
            frame_idx = start_frame + f
            time_val = start_time + f * dt
            
            for i in range(num_drones):
                data_list.append([frame_idx, time_val, "Drone", ids[i], curr_pos[i,0], curr_pos[i,1], curr_pos[i,2], 255, 255, 255])

        df_trans = pd.DataFrame(data_list, columns=["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
        df_trans = TrajectoryOptimizer.apply_physics_repulsion(df_trans, safe_dist)
        return df_trans