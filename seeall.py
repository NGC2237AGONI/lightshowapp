import fbx
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def extract_raw_fbx_points(filepath):
    print(f"正在加载 FBX 文件: {filepath}")
    
    # 1. 初始化 FBX SDK
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "Scene")
    importer = fbx.FbxImporter.Create(manager, "")

    if not importer.Initialize(filepath, -1, manager.GetIOSettings()):
        print(f"错误: 无法打开文件 {filepath}")
        return None
    
    importer.Import(scene)
    importer.Destroy()

    raw_points =[]
    total_vertices = 0

    # 2. 递归遍历所有节点
    def process_node(node):
        nonlocal total_vertices
        attr = node.GetNodeAttribute()
        
        # 只找包含 Mesh (网格) 的节点
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            mesh = node.GetMesh()
            num_verts = mesh.GetControlPointsCount()
            total_vertices += num_verts
            ctrl_points = mesh.GetControlPoints()
            
            # 拿到节点在世界坐标系下的绝对矩阵 (第0帧)
            time_zero = fbx.FbxTime(0)
            global_transform = node.EvaluateGlobalTransform(time_zero)
            
            # 不做任何花里胡哨的抽样，逐个顶点硬读
            for i in range(num_verts):
                pt = ctrl_points[i]
                # 用全局变换矩阵乘以局部顶点，得到世界绝对坐标
                final_pt = global_transform.MultT(pt)
                raw_points.append([final_pt[0], final_pt[1], final_pt[2]])
        
        # 找子节点（比如马腿挂在马肚子下面）
        for i in range(node.GetChildCount()):
            process_node(node.GetChild(i))

    # 从根节点开始找
    root = scene.GetRootNode()
    if root:
        process_node(root)

    manager.Destroy()
    
    points_array = np.array(raw_points)
    print(f"提取完成！FBX 模型内共有绝对原始顶点: {len(points_array)} 个。")
    return points_array


def show_points_in_matplotlib(points):
    if points is None or len(points) == 0:
        print("未提取到任何点，退出渲染。")
        return
        
    print("正在启动 3D 渲染引擎 (若点数超过十万，请耐心等待 3-5 秒)...")
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 初始化纯黑学术风画布
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # 强制等比例坐标轴（防止马被拉长变成腊肠狗）
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 关闭网格和坐标轴，纯粹看点
    ax.axis('off')

    # 用实心小圆点画出来
    # s=0.5 调整点的大小，防止高模点全糊在一起
    ax.scatter(x, y, z, c='white', s=0.5, marker='o', edgecolors='none', alpha=0.8)

    plt.title("Raw FBX Vertices (Sanity Check Tool)", color='white', pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 用简单的弹窗让你选文件
    root = tk.Tk()
    root.withdraw() 
    filepath = filedialog.askopenfilename(
        title="请选择要透视的 FBX 文件", 
        filetypes=[("FBX files", "*.fbx")]
    )
    
    if filepath:
        pts = extract_raw_fbx_points(filepath)
        show_points_in_matplotlib(pts)
    else:
        print("未选择文件。")