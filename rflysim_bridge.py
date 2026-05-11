import pandas as pd
import numpy as np
import time
import socket
import struct
import traceback

class RflySimVisualBridge:
    def __init__(self, ue4_ip='127.0.0.1', ue4_port=20010):
        """
        初始化 RflySim UDP 桥接器
        :param ue4_ip: 运行 RflySim3D (Unreal Engine) 的电脑 IP，单机运行默认为 127.0.0.1
        :param ue4_port: RflySim3D 默认的 UDP 接收端口
        """
        self.ip = ue4_ip
        self.port = ue4_port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 允许端口复用和广播
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        print(f"[桥接器就绪] 目标地址: {self.ip}:{self.port}")

    def send_drone_state(self, drone_id, x, y, z, r, g, b):
        """
        根据 RflySim 的底层 C++ 结构体协议打包数据并发送
        RflySim3D 标准位置包格式 (简易模式):
        Checksum(1234567890), PacketLen, CopterID, X, Y, Z, Roll, Pitch, Yaw
        """
        try:
            # 1. 坐标系转换 (Sim2Real 映射)
            # RflySim/UE4 通常使用 ENU (东北天) 或 NED (北东地)。
            # 假设你的 CSV 是标准的局部右手系 (Z朝上)，我们将其映射到 UE4 的 ENU 坐标系。
            # 如果你在引擎里发现方向反了，只需在这里修改正负号即可。
            ue_x = x
            ue_y = y
            ue_z = z

            # 2. 颜色映射
            # RflySim 中的特殊灯光秀无人机（如 Ghost 模型），通常会将颜色编码到姿态角或预留位中。
            # 这里我们采用通用的 RGB 归一化 (0-1) 预留发送接口。
            # 注：具体灯光控制通道需参考你所选 RflySim 飞机模型的 API 文档。
            color_r = r / 255.0
            color_g = g / 255.0
            color_b = b / 255.0

            # 3. 按照 RflySim3D struct 协议打包 (3个int, 6个float)
            # 校验码为 1234567890, 长度目前简化处理
            checksum = 1234567890
            packet_len = 36 # 简易包长度
            
            # 使用 struct 打包为二进制流 (小端序)
            # 格式: < 3个整数(i) 6个浮点数(f)
            # 发送 位置(X,Y,Z) 和 姿态(暂用RGB颜色位替代，具体视引擎材质蓝图而定)
            buf = struct.pack('<3i6f', 
                              checksum, packet_len, int(drone_id), 
                              ue_x, ue_y, ue_z, 
                              color_r, color_g, color_b)
            
            self.udp_socket.sendto(buf, (self.ip, self.port))
            
        except Exception as e:
            pass # 忽略单帧单架飞机的发送错误，防止阻塞整个集群

def play_csv_in_rflysim(csv_file):
    print("="*50)
    print(" RflySim 虚拟灯光秀引擎桥接程序 ")
    print("="*50)
    
    bridge = RflySimVisualBridge()
    
    try:
        print(f"正在加载轨迹文件: {csv_file} ...")
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['Frame', 'Time', 'VertexID', 'X', 'Y', 'Z'])
        
        # 确保按时间严格排序
        df = df.sort_values(by=['Time', 'VertexID'])
        unique_times = df['Time'].unique()
        unique_times.sort()
        
        print(f"加载成功！共计 {len(unique_times)} 帧数据。")
        print(">> 准备发送，请确保已打开 RflySim3D.exe 并加载了正确的地图和飞机模型。")
        input(">> 按回车键 [Enter] 开始播放...")
        print("播放中...")

        start_time_real = time.time()
        start_time_csv = unique_times[0]

        for current_t in unique_times:
            # 1. 提取当前时刻所有飞机的数据
            frame_data = df[df['Time'] == current_t]
            
            # 2. 遍历并发送 UDP 包
            for _, row in frame_data.iterrows():
                # 注意：RflySim 的 ID 通常从 1 开始，如果你的 VertexID 从 0 开始，建议 +1
                drone_id = int(row['VertexID']) + 1 
                bridge.send_drone_state(
                    drone_id, 
                    row['X'], row['Y'], row['Z'], 
                    row['R'], row['G'], row['B']
                )

            # 3. 极其严谨的同步补偿机制 (防止 time.sleep() 带来的累积误差)
            elapsed_csv = current_t - start_time_csv
            elapsed_real = time.time() - start_time_real
            
            sleep_time = elapsed_csv - elapsed_real
            if sleep_time > 0:
                # 如果代码跑得比 CSV 设定的时间快，就稍微等一下
                time.sleep(sleep_time)

        print("\n✅ 轨迹播放结束！")
        
    except Exception as e:
        print(f"\n❌ 播放中断或发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 在这里填入你的 CSV 文件路径
    TARGET_CSV = "Full_Show.csv"
    play_csv_in_rflysim(TARGET_CSV)