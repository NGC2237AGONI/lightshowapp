import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys


INPUT_CSV = "drone_path.csv"   # 轨迹文件
PLAY_SPEED = 0.75               # 播放速度
POINT_SIZE = 10                # 点大小 


def load_drone_paths(csv_file):
    print(f"读取彩色动画数据: {csv_file} ...")
    frames = {}
    all_x, all_y, all_z = [], [], []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                frame_idx = int(row[0])
                x, y, z = float(row[4]), float(row[5]), float(row[6])
                
                # 读取颜色列 (包括旧CSV)
                try:
                    r = float(row[7]) / 255.0
                    g = float(row[8]) / 255.0
                    b = float(row[9]) / 255.0
                except IndexError:
                    r, g, b = 1.0, 1.0, 1.0 

                if frame_idx not in frames: frames[frame_idx] = []
                
                # 存入 [x, y, z, r, g, b]
                frames[frame_idx].append([x, y, z, r, g, b])
                
                if frame_idx % 10 == 0: 
                    all_x.append(x); all_y.append(y); all_z.append(z)
    except FileNotFoundError:
        print("错误：找不到文件。")
        sys.exit()

    print(f"共 {len(frames)} 帧动画。")
    
    if not all_x: return {}, {}
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    min_z, max_z = min(all_z), max(all_z)
    
    max_range = max([max_x-min_x, max_y-min_y, max_z-min_z]) / 2.0
    mid_x, mid_y, mid_z = (max_x+min_x)/2, (max_y+min_y)/2, (max_z+min_z)/2
    
    limits = {
        'x': (mid_x - max_range, mid_x + max_range),
        'y': (mid_y - max_range, mid_y + max_range),
        'z': (mid_z - max_range, mid_z + max_range)
    }
    return frames, limits

def main():
    frames_data, limits = load_drone_paths(INPUT_CSV)
    if not frames_data: return

    sorted_indices = sorted(frames_data.keys())
    total_frames = len(sorted_indices)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black'); fig.patch.set_facecolor('black')
    ax.axis('off')
    
    scat = ax.scatter([], [], [], c='white', s=POINT_SIZE, alpha=1.0, edgecolors='none')
    txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='white')

    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    ax.set_zlim(limits['z'])
    try: ax.set_box_aspect([1, 1, 1])
    except: pass

    print("循环播放ing")

    def update(frame_num):
        idx = frame_num % total_frames
        real_frame = sorted_indices[idx]
        
        # data结构: [N, 6] -> (x, y, z, r, g, b)
        data = np.array(frames_data[real_frame])
        
        # 更新位置
        scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        # 更新颜色
        scat.set_color(data[:, 3:6])
        
        txt.set_text(f"Frame: {idx+1}/{total_frames}")
        return scat, txt

    interval = (1000 / 20) / PLAY_SPEED
    ani = animation.FuncAnimation(fig, update, interval=interval, blit=False, cache_frame_data=False)
    
    plt.show()

if __name__ == "__main__":
    main()