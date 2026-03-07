import fbx
import os
from PIL import Image

# 全局缓存
_TEXTURE_CACHE = {}

def build_texture_cache(root_dir):
    """
    建立 {无后缀文件名: 完整路径} 的映射，用于快速查找
    """
    if root_dir in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[root_dir]
    
    # 这一句如果打印出来了，说明你用的是新代码
    print(f"   [系统] 正在建立纹理索引: {root_dir}")
    cache = {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.tga', '.bmp', '.tif', '.tiff'}
    
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                full_path = os.path.join(root, f)
                # 键只存小写的无后缀名，例如 "horse_diffuse"
                basename = os.path.splitext(f)[0].lower() 
                if basename not in cache:
                    cache[basename] = full_path
                
    _TEXTURE_CACHE[root_dir] = cache
    return cache

def find_texture_smart(fbx_path, texture_name):
    # 确定搜索根目录：FBX目录的上一级的上一级
    fbx_dir = os.path.dirname(os.path.abspath(fbx_path))
    search_root = os.path.dirname(fbx_dir)
    if len(search_root) < 3: search_root = fbx_dir # 防御
    
    cache = build_texture_cache(search_root)
    
    # 目标文件名也去掉后缀
    target_basename = os.path.splitext(os.path.basename(texture_name))[0].lower()
    
    # 1. 精确匹配 (忽略后缀)
    if target_basename in cache:
        print(f"   [纹理匹配] {texture_name} -> {os.path.basename(cache[target_basename])}")
        return cache[target_basename]
        
    # 2. 包含匹配 (应对 _CRS, _D 等后缀差异)
    # 查找 cache 中是否包含 target_basename，或者反之
    for key, path in cache.items():
        if target_basename in key or key in target_basename:
            print(f"   [模糊匹配] {texture_name} -> {os.path.basename(path)}")
            return path

    return None

def get_texture_colors(mesh, node, base_path="."):
    num_verts = mesh.GetControlPointsCount()
    default_color = (0.5, 0.5, 0.5)
    final_colors = [default_color] * num_verts

    # 1. 寻找纹理名
    texture_fbx_path = None
    if node.GetMaterialCount() > 0:
        for m_idx in range(node.GetMaterialCount()):
            mat = node.GetMaterial(m_idx)
            if not mat: continue
            prop = mat.GetFirstProperty()
            while prop.IsValid():
                if prop.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxFileTexture.ClassId)) > 0:
                    tex = prop.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxFileTexture.ClassId), 0)
                    if tex:
                        texture_fbx_path = tex.GetFileName()
                        break 
                prop = mat.GetNextProperty(prop)
            if texture_fbx_path: break
    
    if not texture_fbx_path: return None 

    # 2. 智能搜索
    real_image_path = find_texture_smart(os.path.join(base_path, "placeholder.fbx"), texture_fbx_path)

    if not real_image_path:
        print(f"   [警告] 找不到贴图: {os.path.basename(texture_fbx_path)} (智能搜索失败)")
        return None

    # 3. 加载图片
    try:
        img = Image.open(real_image_path)
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024))
        img_width, img_height = img.size
        rgb_img = img.convert('RGB') 
    except Exception as e:
        print(f"   [错误] 图片加载失败: {e}")
        return None

    # 4. UV 采样
    uv_layer = mesh.GetElementUV(0)
    if not uv_layer: return None
        
    uv_direct = uv_layer.GetDirectArray()
    uv_index = uv_layer.GetIndexArray()
    mapping_mode = uv_layer.GetMappingMode()
    ref_mode = uv_layer.GetReferenceMode()

    vert_uv_map = {} 

    if mapping_mode == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
        poly_count = mesh.GetPolygonCount()
        vertex_counter = 0
        for i in range(poly_count):
            poly_size = mesh.GetPolygonSize(i)
            for j in range(poly_size):
                control_point_idx = mesh.GetPolygonVertex(i, j)
                if control_point_idx not in vert_uv_map:
                    uv_idx = vertex_counter
                    if ref_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                        uv_idx = uv_index.GetAt(vertex_counter)
                    uv = uv_direct.GetAt(uv_idx)
                    vert_uv_map[control_point_idx] = uv
                vertex_counter += 1     
    elif mapping_mode == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
        for i in range(num_verts):
            uv_idx = i
            if ref_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                uv_idx = uv_index.GetAt(i)
            uv = uv_direct.GetAt(uv_idx)
            vert_uv_map[i] = uv

    for i in range(num_verts):
        if i in vert_uv_map:
            u, v = vert_uv_map[i]
            u = u - int(u); v = v - int(v)
            if u < 0: u += 1
            if v < 0: v += 1
            x = int(u * (img_width - 1))
            y = int((1.0 - v) * (img_height - 1)) 
            pixel = rgb_img.getpixel((x, y))
            final_colors[i] = (pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0)
            
    return final_colors