import fbx
import sys
import csv
import numpy as np


INPUT_FILE = "3.fbx"           
FORMATION_FILE = "final_formation.npz" 
OUTPUT_FILE = "drone_path.csv"   # 结果
FPS = 20                         # 帧率
SCALE = 2.0                      # 基础缩放 
AXIS_MODE = 1                    # 坐标修正 

# 判定静止的阈值 (米)
STATIC_THRESHOLD = 0.002 
# 尾部缓冲时间 (秒)
END_BUFFER_TIME = 0.5

# 全局变量
TARGET_MAP = {}
COLOR_MAP = {} 
SKINNING_DATA = {} 

def main():
    print(f"读取提取编队名单: {FORMATION_FILE}...")
    try:
        data = np.load(FORMATION_FILE)
        names = data['mesh_names']
        ids = data['vertex_ids']
        colors = data['ref_colors'] 
        
        for n, i, c in zip(names, ids, colors):
            n_str = str(n)
            key = (n_str, int(i))
            
            if n_str not in TARGET_MAP: TARGET_MAP[n_str] = set()
            TARGET_MAP[n_str].add(int(i))
            
            COLOR_MAP[key] = c
            
        print(f"加载完成，共 {len(names)} 个点")
    except FileNotFoundError:
        print("错误：找不到 final_formation.npz")
        return

    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "Scene")
    importer = fbx.FbxImporter.Create(manager, "")

    print(f"读取模型: {INPUT_FILE}...")
    if not importer.Initialize(INPUT_FILE, -1, manager.GetIOSettings()): return
    importer.Import(scene)
    importer.Destroy()

    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
    num_stacks = scene.GetSrcObjectCount(criteria)
    print("\n 可选动作:")
    stacks = []
    for i in range(num_stacks):
        s = scene.GetSrcObject(criteria, i)
        stacks.append(s)
        span = s.GetLocalTimeSpan()
        dur = span.GetStop().GetSecondDouble() - span.GetStart().GetSecondDouble()
        print(f"  [{i}] {s.GetName()} (原始时长: {dur:.1f}s)")
    
    if num_stacks > 0:
        try:
            sel_str = input(f"选择动作 (0-{num_stacks-1}): ")
            sel = int(sel_str)
            target_stack = stacks[sel]
            scene.SetCurrentAnimationStack(target_stack)
            
            span = target_stack.GetLocalTimeSpan()
            start = span.GetStart().GetSecondDouble()
            end = span.GetStop().GetSecondDouble()
            max_frames = int((end - start) * FPS)
        except:
            print("输入无效，导出默认。")
            max_frames = 1
            start = 0
    else:
        print("无动画。")
        max_frames = 1
        start = 0

    print("解析骨骼蒙皮数据")
    prepare_skinning_data(scene)

    print(f"分析轨迹 (FPS={FPS})...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Time", "Object", "VertexID", "X", "Y", "Z", "R", "G", "B"])
        
        root = scene.GetRootNode()
        if root:
            process_node_smart(root, start, max_frames, writer)
            
    print(f"\n 运行成功，已导出至 {OUTPUT_FILE}")
    manager.Destroy()

def prepare_skinning_data(scene):
    global SKINNING_DATA
    SKINNING_DATA = {} 
    src_len = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId))
    for i in range(src_len):
        mesh = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxMesh.ClassId), i)
        node = mesh.GetNode()
        if not node: continue
        mesh_name = node.GetName()
        if mesh_name not in TARGET_MAP: continue
        target_indices = TARGET_MAP[mesh_name]
        
        skin_deformer = None
        deformer_count = mesh.GetDeformerCount()
        for j in range(deformer_count):
            df = mesh.GetDeformer(j)
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
            num_indices = cluster.GetControlPointIndicesCount()
            
            for k in range(num_indices):
                v_idx = indices[k]
                w = weights[k]
                if v_idx in target_indices:
                    key = (mesh_name, v_idx)
                    if key not in SKINNING_DATA: SKINNING_DATA[key] = []
                    SKINNING_DATA[key].append( (bone_node, bind_matrix_inv, w) )

def process_node_smart(node, start_time, max_frames, writer):
    attr = node.GetNodeAttribute()
    if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
        mesh_name = node.GetName()
        if mesh_name in TARGET_MAP:
            extract_and_prune_data(node, mesh_name, start_time, max_frames, writer)
            
    for i in range(node.GetChildCount()):
        process_node_smart(node.GetChild(i), start_time, max_frames, writer)

def extract_and_prune_data(node, mesh_name, start_time, max_frames, writer):
    mesh = node.GetMesh()
    target_ids = TARGET_MAP[mesh_name]
    local_verts = mesh.GetControlPoints()
    fbx_time = fbx.FbxTime()
    
    # 检查骨骼
    has_skin = False
    test_key = (mesh_name, list(target_ids)[0])
    if test_key in SKINNING_DATA: has_skin = True
    
    print(f"Mesh: {mesh_name} (最大 {max_frames} 帧) ")

    # 缓存数据
    frame_buffer = [] 
    prev_frame_coords = None
    last_moving_frame = 0 

    for f in range(max_frames + 1):
        curr_sec = start_time + (f / FPS)
        fbx_time.SetSecondDouble(curr_sec)
        global_trans = node.EvaluateGlobalTransform(fbx_time)
        
        current_frame_data = [] 
        current_coords_only = [] 

        for v_idx in target_ids:
            local_pos = local_verts[v_idx]
            final_vec = fbx.FbxVector4(0, 0, 0, 0)
            
            if has_skin:
                skin_info = SKINNING_DATA.get((mesh_name, v_idx), [])
                if not skin_info:
                    final_vec = global_trans.MultT(local_pos)
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
            
            # 缩放
            x = final_vec[0] * SCALE
            y = final_vec[1] * SCALE
            z = final_vec[2] * SCALE
            
            # 坐标修正
            if AXIS_MODE == 1: x, y, z = x, z, y
            elif AXIS_MODE == 2: x, y, z = z, y, x
            elif AXIS_MODE == 3: x, y, z = y, x, z
            elif AXIS_MODE == 4: y, z = z, -y
            
            #获取颜色并打包
            rgb = COLOR_MAP.get((mesh_name, v_idx), [1.0, 1.0, 1.0])
            r = int(rgb[0] * 255)
            g = int(rgb[1] * 255)
            b = int(rgb[2] * 255)
            
            #存入缓存
            current_frame_data.append( (v_idx, x, y, z, curr_sec, r, g, b) )
            current_coords_only.append( [x, y, z] )
        
        frame_buffer.append(current_frame_data)
        
        #运动检测
        curr_coords_np = np.array(current_coords_only)
        
        if prev_frame_coords is not None:
            movements = np.linalg.norm(curr_coords_np - prev_frame_coords, axis=1)
            max_move = np.max(movements)
            if max_move > STATIC_THRESHOLD:
                last_moving_frame = f
                
        prev_frame_coords = curr_coords_np

    #智能截断
    buffer_frames = int(END_BUFFER_TIME * FPS)
    final_end_frame = min(last_moving_frame + buffer_frames, max_frames)
    final_end_frame = max(final_end_frame, 1) 
    
    saved_time = (max_frames - final_end_frame) / FPS
    print(f"      [智能检测] 原始: {max_frames}帧 -> 有效: {final_end_frame}帧 (剪掉 {saved_time:.2f}s)")
    
    # 写入CSV
    for f in range(final_end_frame):
        frame_data_list = frame_buffer[f]
        for item in frame_data_list:
            # 解包颜色
            v_idx, x, y, z, sec, r, g, b = item
            writer.writerow([f, f"{sec:.3f}", mesh_name, v_idx, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", r, g, b])

if __name__ == "__main__":
    main()