import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from drone_core import TrajectoryOptimizer
from config_manager import cfg
import traceback
import math

class CompositionManager:
    def __init__(self):
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
                'position': [0, 0, 0],
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

    def set_position(self, index, x, y, z):
        if 0 <= index < len(self.playlist):
            self.playlist[index]['position'] = [x, y, z]

    def _apply_transform(self, df, rotation, position):
        if not rotation and not position:
            return 
            
        first_frame = df['Frame'].min()
        first_frame_pts = df[df['Frame'] == first_frame][['X', 'Y', 'Z']].values
        pivot = np.mean(first_frame_pts, axis=0)
        
        pts = df[['X', 'Y', 'Z']].values
        pts_centered = pts - pivot
        
        if any(r != 0 for r in rotation):
            rads = np.radians(rotation)
            rx, ry, rz = rads[0], rads[1], rads[2]
            
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            
            R = Rz @ Ry @ Rx
            pts_centered = pts_centered @ R.T
            
        final_pts = pts_centered + pivot + np.array(position)
        df['X'] = final_pts[:, 0]
        df['Y'] = final_pts[:, 1]
        df['Z'] = final_pts[:, 2]

    def _enforce_boundaries(self, df, bounds):
        L, W, H = bounds
        min_x, max_x = df['X'].min(), df['X'].max()
        min_y, max_y = df['Y'].min(), df['Y'].max()
        min_z, max_z = df['Z'].min(), df['Z'].max()
        
        shift_x = 0; shift_y = 0; shift_z = 0
        if max_x > L/2: shift_x = (L/2) - max_x
        elif min_x < -L/2: shift_x = (-L/2) - min_x
        if max_y > W/2: shift_y = (W/2) - max_y
        elif min_y < -W/2: shift_y = (-W/2) - min_y
        if max_z > H: shift_z = H - max_z 
        elif min_z < 0: shift_z = 0 - min_z
        
        if shift_x != 0 or shift_y != 0 or shift_z != 0:
            df['X'] += shift_x
            df['Y'] += shift_y
            df['Z'] += shift_z

    def _generate_grid(self, num_drones, spacing=2.5):
        side = int(math.ceil(math.sqrt(num_drones)))
        grid_pts = []
        offset = (side - 1) * spacing / 2.0
        for i in range(num_drones):
            r = i // side
            c = i % side
            grid_pts.append([r * spacing - offset, c * spacing - offset, 0.0])
        return np.array(grid_pts)

    def _generate_cube(self, num_drones, center_z, spacing=3.5):
        side = int(math.ceil(num_drones ** (1/3)))
        cube_pts = []
        offset = (side - 1) * spacing / 2.0
        for i in range(num_drones):
            z = i // (side * side)
            rem = i % (side * side)
            r = rem // side
            c = rem % side
            cube_pts.append([r * spacing - offset, c * spacing - offset, center_z + (z * spacing - offset)])
        return np.array(cube_pts)

    # 【新增核心引擎】：严格推导物理底线时间
    def _calc_safe_duration(self, max_dist, max_vel, max_acc):
        if max_dist < 1e-4: return 2.0
        t_min_v = (math.pi * max_dist) / (2.0 * max_vel)
        t_min_a = math.pi * math.sqrt(max_dist / (2.0 * max_acc))
        # 统一 1.05 倍安全冗余，向上取整至 0.1s
        return float(np.ceil(max(2.0, max(t_min_v, t_min_a) * 1.05) * 10) / 10.0)

    def _generate_transition_data(self, start_pos, end_pos, frames, dt, start_frame, start_time, ids, safe_dist):
        num_drones = len(start_pos)
        data_list = []
        np.random.seed(42) 
        arc_heights = np.random.uniform(2.0, 12.0, size=num_drones)
        for f in range(1, frames + 1): 
            t = f / float(frames + 1) 
            factor = 0.5 - 0.5 * np.cos(np.pi * t)
            curr_pos = start_pos + (end_pos - start_pos) * factor
            z_lift = arc_heights * np.sin(np.pi * t)
            curr_pos[:, 2] += z_lift
            frame_idx = start_frame + f
            time_val = start_time + f * dt
            for i in range(num_drones):
                # 统一打上 Sys_Transition 标签供播放器识别
                data_list.append([frame_idx, time_val, "Sys_Transition", ids[i], curr_pos[i,0], curr_pos[i,1], curr_pos[i,2], 255, 255, 255])
        df_trans = pd.DataFrame(data_list, columns=["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
        df_trans = TrajectoryOptimizer.apply_physics_repulsion(df_trans, safe_dist)
        return df_trans

    def merge_shows(self, output_path, safe_dist, bounds, max_vel):
        if len(self.playlist) < 1: return False, "请至少添加一个文件"
        full_df = pd.DataFrame(); fps = cfg.high_density_fps; dt = 1.0 / fps
        L, W, H = bounds; center_z = H / 2.0
        
        try:
            first_item = self.playlist[0]
            num_drones = len(first_item['data'][first_item['data']['Frame'] == first_item['data']['Frame'].min()])
            grid_spacing = max(2.0, safe_dist * 1.5)
            cube_spacing = max(3.0, safe_dist * 2.0)
            
            ground_grid = self._generate_grid(num_drones, spacing=grid_spacing)
            ready_cube = self._generate_cube(num_drones, center_z, spacing=cube_spacing)
            curr_f = 0; curr_t = 0.0

            # ========================================================
            # 1. 起飞 (基于全局 max_vel 严格推导起飞时长)
            # ========================================================
            max_dist_takeoff = np.max(np.linalg.norm(ready_cube - ground_grid, axis=1))
            takeoff_dur = self._calc_safe_duration(max_dist_takeoff, max_vel, cfg.max_accel)
            takeoff_frames = int(takeoff_dur * fps)
            takeoff_data = []
            
            for f in range(takeoff_frames + 1):
                interp = 0.5 - 0.5 * np.cos(np.pi * (f/takeoff_frames)) if takeoff_frames > 0 else 1.0
                curr_pos = ground_grid + (ready_cube - ground_grid) * interp
                for i in range(num_drones):
                    takeoff_data.append([curr_f + f, curr_t + f*dt, "Sys_Takeoff", i, curr_pos[i,0], curr_pos[i,1], curr_pos[i,2], 255, 255, 255])
                    
            full_df = pd.DataFrame(takeoff_data, columns=["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
            curr_f += takeoff_frames; curr_t += takeoff_dur
            
            physical_pos = ready_cube.copy(); physical_ids = np.arange(num_drones)

            # ========================================================
            # 2. 表演循环
            # ========================================================
            for i in range(len(self.playlist)):
                item = self.playlist[i]
                df_item = item['data'].copy()
                self._apply_transform(df_item, item.get('rotation', [0,0,0]), item.get('position', [0,0,0]))
                self._enforce_boundaries(df_item, bounds)
                
                start_data = df_item[df_item['Frame'] == df_item['Frame'].min()].sort_values('VertexID')
                target_pos_raw = start_data[['X', 'Y', 'Z']].values
                target_ids_raw = start_data['VertexID'].values
                
                dist_matrix = cdist(physical_pos, target_pos_raw)
                _, col_ind = linear_sum_assignment(dist_matrix)
                
                id_mapping = {target_ids_raw[col_ind[pid]]: pid for pid in range(num_drones)}
                df_item['VertexID'] = df_item['VertexID'].map(id_mapping)
                df_item = df_item.sort_values(by=['Frame', 'VertexID'])
                
                # 基于全局 max_vel 严格推导过渡时长
                max_dist = np.max(np.linalg.norm(physical_pos - target_pos_raw[col_ind], axis=1))
                physics_dur = self._calc_safe_duration(max_dist, max_vel, cfg.max_accel)
                trans_dur = max(item['transition_dur'], physics_dur)
                
                trans_df = self._generate_transition_data(physical_pos, target_pos_raw[col_ind], int(trans_dur * fps), dt, curr_f, curr_t, physical_ids, safe_dist)
                full_df = pd.concat([full_df, trans_df], ignore_index=True)
                curr_f = full_df['Frame'].max(); curr_t = full_df['Time'].max()

                # 到位定格 (1.0s)
                wait_dur = 1.0 
                wait_frames = int(wait_dur * fps)
                wait_data = []
                for w_f in range(1, wait_frames + 1):
                    for pid in range(num_drones):
                        wait_data.append([curr_f + w_f, curr_t + w_f*dt, "Sys_Wait", pid, target_pos_raw[col_ind][pid,0], target_pos_raw[col_ind][pid,1], target_pos_raw[col_ind][pid,2], 255, 255, 255])
                full_df = pd.concat([full_df, pd.DataFrame(wait_data, columns=full_df.columns)], ignore_index=True)
                curr_f += wait_frames; curr_t += wait_dur

                # 拼接表演动画
                df_item['Time'] = df_item['Time'] - df_item['Time'].min() + curr_t + dt
                df_item['Frame'] = df_item['Frame'] - df_item['Frame'].min() + curr_f + 1
                full_df = pd.concat([full_df, df_item], ignore_index=True)
                curr_f = full_df['Frame'].max(); curr_t = full_df['Time'].max()
                physical_pos = df_item[df_item['Frame'] == curr_f].sort_values('VertexID')[['X', 'Y', 'Z']].values

            # ========================================================
            # 3. 返航 (基于全局 max_vel 严格推导)
            # ========================================================
            dist_matrix_back = cdist(physical_pos, ready_cube)
            _, col_ind_back = linear_sum_assignment(dist_matrix_back)
            
            max_dist_back = np.max(np.linalg.norm(physical_pos - ready_cube[col_ind_back], axis=1))
            back_dur = self._calc_safe_duration(max_dist_back, max_vel, cfg.max_accel)
            
            back_df = self._generate_transition_data(physical_pos, ready_cube[col_ind_back], int(back_dur * fps), dt, curr_f, curr_t, physical_ids, safe_dist)
            full_df = pd.concat([full_df, back_df], ignore_index=True)
            curr_f = full_df['Frame'].max(); curr_t = full_df['Time'].max()

            # ========================================================
            # 4. 降落 (基于全局 max_vel 严格推导)
            # ========================================================
            land_target = ground_grid[col_ind_back]
            max_dist_land = np.max(np.linalg.norm(ready_cube[col_ind_back] - land_target, axis=1))
            land_dur = self._calc_safe_duration(max_dist_land, max_vel, cfg.max_accel)
            
            land_frames = int(land_dur * fps)
            land_data = []
            for f in range(1, land_frames + 1):
                interp = 0.5 - 0.5 * np.cos(np.pi * (f/land_frames))
                curr_pos = ready_cube[col_ind_back] + (land_target - ready_cube[col_ind_back]) * interp
                for pid in range(num_drones):
                    land_data.append([curr_f + f, curr_t + f*dt, "Sys_Land", pid, curr_pos[pid,0], curr_pos[pid,1], curr_pos[pid,2], 255, 255, 255])
            
            full_df = pd.concat([full_df, pd.DataFrame(land_data, columns=full_df.columns)], ignore_index=True)

            full_df[["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"]].to_csv(output_path, index=False)
            return True, f"编队闭环合成成功！物理总时长: {full_df['Time'].max():.2f}s"
        except Exception as e:
            traceback.print_exc()
            return False, f"合成出错: {str(e)}"

    def get_min_safe_transition_time(self, index, bounds, max_vel):
        if index < 0 or index >= len(self.playlist) - 1: return 1.0 
        try:
            item_curr = self.playlist[index]
            df_curr = item_curr['data'].copy()
            self._apply_transform(df_curr, item_curr.get('rotation', [0,0,0]), item_curr.get('position', [0,0,0]))
            self._enforce_boundaries(df_curr, bounds)
            p1 = df_curr[df_curr['Frame'] == df_curr['Frame'].max()].sort_values('VertexID')[['X', 'Y', 'Z']].values
            
            item_next = self.playlist[index + 1]
            df_next = item_next['data'].copy()
            self._apply_transform(df_next, item_next.get('rotation', [0,0,0]), item_next.get('position', [0,0,0]))
            self._enforce_boundaries(df_next, bounds)
            p2 = df_next[df_next['Frame'] == df_next['Frame'].min()].sort_values('VertexID')[['X', 'Y', 'Z']].values
            
            dist = np.max(np.linalg.norm(p1 - p2[linear_sum_assignment(cdist(p1, p2))[1]], axis=1))
            return self._calc_safe_duration(dist, max_vel, cfg.max_accel)
        except: return 1.0