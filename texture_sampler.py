import fbx
import os
from PIL import Image # 需要 pip install Pillow

def get_texture_colors(mesh, node, base_path="."):
    """
    [高级功能] 从贴图(Texture)中采样颜色
    """
    num_verts = mesh.GetControlPointsCount()
    default_color = (0.5, 0.5, 0.5)
    final_colors = [default_color] * num_verts

    # 1. 寻找纹理文件路径
    texture_filename = None
    
    # 遍历材质寻找贴图
    if node.GetMaterialCount() > 0:
        mat = node.GetMaterial(0)
        # 查找 Diffuse 或 BaseColor 属性上的纹理
        for prop_name in ["DiffuseColor", "BaseColor", "Color"]:
            prop = mat.FindProperty(prop_name)
            if prop.IsValid():
                # 获取连接到该属性的纹理数量
                tex_count = prop.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxFileTexture.ClassId))
                if tex_count > 0:
                    # 获取第一个纹理
                    texture = prop.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxFileTexture.ClassId), 0)
                    texture_filename = texture.GetFileName()
                    break
    
    if not texture_filename:
        print("   [警告] 未在材质中找到关联的贴图文件。")
        return None # 返回 None 表示没找到贴图，请回退到材质色逻辑

    # 2. 处理路径 (处理绝对路径和相对路径问题)
    # 很多 FBX 里的路径是绝对路径（如 C:\Users\Artist\Desktop...）
    # 我们优先在当前文件夹找同名图片
    basename = os.path.basename(texture_filename)
    local_image_path = os.path.join(base_path, basename)
    
    real_image_path = ""
    if os.path.exists(local_image_path):
        real_image_path = local_image_path
    elif os.path.exists(texture_filename):
        real_image_path = texture_filename
    else:
        print(f"   [错误] 找不到贴图文件: {basename}")
        print(f"          请确保图片文件和 FBX 在同一个文件夹！")
        return None

    print(f"   [图片] 正在读取贴图: {real_image_path}")

    # 3. 加载图片
    try:
        img = Image.open(real_image_path)
        img_width, img_height = img.size
        rgb_img = img.convert('RGB') # 确保是 RGB 模式
    except Exception as e:
        print(f"   [错误] 图片加载失败: {e}")
        return None

    # 4. 获取 UV 坐标
    # UV 通常存储在 Layer 0
    uv_layer = mesh.GetElementUV(0)
    if not uv_layer:
        print("   [错误] 模型没有 UV 坐标，无法读取贴图颜色。")
        return None
        
    uv_direct = uv_layer.GetDirectArray()
    uv_index = uv_layer.GetIndexArray()
    mapping_mode = uv_layer.GetMappingMode()
    ref_mode = uv_layer.GetReferenceMode()

    # 5. 遍历每个顶点，采样颜色
    print("   [处理] 正在对 贴图 进行采样...")
    
    # 为了简化，我们假设 Mesh 是三角化的，且映射模式是 ByPolygonVertex (最常见)
    # 或者是 ByControlPoint。这里做一个通用采样。
    
    # 注意：一个顶点可能对应多个 UV (接缝处)，我们简单取第一个关联的 UV
    # 这种方法对于点阵显示（无人机）精度足够了
    
    # 我们需要一个 顶点索引 -> UV索引 的映射
    # FBX 的 UV 存储稍微有点麻烦，我们用一种简化策略：
    # 遍历所有多边形，记录每个顶点最后一次出现的 UV
    
    vert_uv_map = {} # { vert_index : (u, v) }

    # 获取多边形数据来对齐 UV
    if mapping_mode == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
        poly_count = mesh.GetPolygonCount()
        vertex_counter = 0
        for i in range(poly_count):
            poly_size = mesh.GetPolygonSize(i)
            for j in range(poly_size):
                # 获取该多边形角的顶点 ID
                control_point_idx = mesh.GetPolygonVertex(i, j)
                
                # 获取 UV 索引
                uv_idx = vertex_counter
                if ref_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                    uv_idx = uv_index.GetAt(vertex_counter)
                
                # 获取 UV 值
                uv = uv_direct.GetAt(uv_idx)
                
                # 存入映射
                vert_uv_map[control_point_idx] = uv
                
                vertex_counter += 1
                
    elif mapping_mode == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
        for i in range(num_verts):
            uv_idx = i
            if ref_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                uv_idx = uv_index.GetAt(i)
            uv = uv_direct.GetAt(uv_idx)
            vert_uv_map[i] = uv

    # 6. 执行采样
    for i in range(num_verts):
        if i in vert_uv_map:
            u, v = vert_uv_map[i]
            
            # 处理 UV 循环和翻转
            # FBX 的 UV 原点通常在左下角，PIL 在左上角，且可能超过 1.0
            u = u % 1.0
            v = v % 1.0
            
            # 映射到像素坐标
            x = int(u * img_width)
            y = int((1.0 - v) * img_height) # Y轴翻转
            
            # 防止越界
            x = min(x, img_width - 1)
            y = min(y, img_height - 1)
            
            # 获取像素颜色
            pixel = rgb_img.getpixel((x, y))
            final_colors[i] = (pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0)
            
    return final_colors