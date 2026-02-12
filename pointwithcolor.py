import fbx
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import texture_sampler 
import numpy as np


INPUT_FILE = "3.fbx"   
SCALE = 2.0            # 缩放系数
POINT_SIZE = 0.5       

all_x = []
all_y = []
all_z = []
all_c = [] 
all_mesh_names = [] 
all_vertex_ids = [] 

def main():
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "Scene")
    importer = fbx.FbxImporter.Create(manager, "")

    print(f"读取文件: {INPUT_FILE}...")
    if not importer.Initialize(INPUT_FILE, -1, manager.GetIOSettings()):
        print(f"无法打开文件 {INPUT_FILE}")
        return
    importer.Import(scene)
    importer.Destroy()

    print("正在提取初始时刻数据...")
    root = scene.GetRootNode()
    if root:
        process_node(root)

    manager.Destroy()

    if len(all_x) == 0:
        print("未提取到数据。")
        return
    
    print(f"提取完成，共 {len(all_x)} 个点。")
    
    # 保存数据
    points = np.column_stack((all_x, all_y, all_z))
    colors = np.array(all_c)
    mesh_names = np.array(all_mesh_names)
    vertex_ids = np.array(all_vertex_ids)
    
    output_filename = "model_data.npz"
    print(f"保存全量数据到 {output_filename} ...")
    np.savez(output_filename, 
             points=points, 
             colors=colors, 
             mesh_names=mesh_names, 
             vertex_ids=vertex_ids  
             )
    print("数据保存完成。")

def process_node(node):
    attr = node.GetNodeAttribute()
    if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
        extract_mesh_data(node)
    for i in range(node.GetChildCount()):
        process_node(node.GetChild(i))

def extract_mesh_data(node):
    mesh = node.GetMesh()
    num_verts = mesh.GetControlPointsCount()
    mesh_name = node.GetName() # 获取网格名字
    
    colors = texture_sampler.get_texture_colors(mesh, node, base_path=".") 
    if colors is None:
        colors = get_vertex_colors(mesh, node)
    
    local_vertices = mesh.GetControlPoints()
    time_zero = fbx.FbxTime(0)
    global_transform = node.EvaluateGlobalTransform(time_zero)
    
    for v_idx in range(num_verts):
        local_pos = local_vertices[v_idx]
        final_pos = global_transform.MultT(local_pos)
        
        all_x.append(final_pos[0] * SCALE)
        all_y.append(final_pos[1] * SCALE)
        all_z.append(final_pos[2] * SCALE)
        all_c.append(colors[v_idx])
        
        all_mesh_names.append(mesh_name)
        all_vertex_ids.append(v_idx)

def get_vertex_colors(mesh, node):
    num_verts = mesh.GetControlPointsCount()
    default_color = (0.5, 0.5, 0.5) 
    final_colors = [default_color] * num_verts 
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

if __name__ == "__main__":
    if len(sys.argv) > 1: INPUT_FILE = sys.argv[1]
    main()