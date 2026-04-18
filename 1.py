import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ================= 数据预备 =================
metrics = {'Saliency':[21.2, 24.3], 'Chamfer':[6.9298, 5.8208], 'Safety': [1.2000, 5.7198]}

color_old = '#8C92ac'   
color_new_r = '#d62728' 
color_new_g = '#2ca02c' 
color_new_b = '#1f77b4' 

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 

# ================= 2x2 极简四宫格布局 =================
fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor='#f4f4f9')

fig.suptitle('基于格式塔视觉显著性混合采样的消融实验多维性能评估', 
             fontsize=22, fontweight='bold', y=0.96)

# [0, 0] 特征捕获率
axs[0, 0].bar(['传统体素降采样', '视觉感知混合采样\n(Ours)'], 
              metrics['Saliency'], color=[color_old, color_new_r], width=0.4, edgecolor='black', alpha=0.85)
axs[0, 0].set_ylim(0, 30)
axs[0, 0].set_ylabel('显著性特征命中率 (%)', fontweight='bold', fontsize=12)
axs[0, 0].set_title('(A) 微观特征捕捉能力', fontsize=14, fontweight='bold')
axs[0, 0].text(0, metrics['Saliency'][0]+0.5, f"{metrics['Saliency'][0]}%", ha='center', fontsize=13, fontweight='bold')
axs[0, 0].text(1, metrics['Saliency'][1]+0.5, f"{metrics['Saliency'][1]}%", ha='center', color=color_new_r, fontsize=14, fontweight='bold')
axs[0, 0].text(0.5, 27, "▲ +14.6% 相对提升", ha='center', color=color_new_r, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor=color_new_r))

# [0, 1] 倒角误差距离
axs[0, 1].bar(['传统体素降采样', '视觉感知混合采样\n(Ours)'], 
              metrics['Chamfer'], color=[color_old, color_new_g], width=0.4, edgecolor='black', alpha=0.85)
axs[0, 1].set_ylim(0, 8)
axs[0, 1].set_ylabel('倒角几何逼近误差 (Chamfer Dist)', fontweight='bold', fontsize=12)
axs[0, 1].set_title('(B) 宏观外壳几何保真度', fontsize=14, fontweight='bold')
axs[0, 1].text(0, metrics['Chamfer'][0]+0.2, f"{metrics['Chamfer'][0]:.2f}", ha='center', fontsize=13, fontweight='bold')
axs[0, 1].text(1, metrics['Chamfer'][1]+0.2, f"{metrics['Chamfer'][1]:.2f}", ha='center', color=color_new_g, fontsize=14, fontweight='bold')
axs[0, 1].text(0.5, 7.2, "▼ -16.0% 误差下降", ha='center', color=color_new_g, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor=color_new_g))

# [1, 0] 物理安全限界 (已删除冗余文字，全系统一置顶数值)
safe_limit = 1.5
axs[1, 0].bar(['传统体素降采样', '视觉感知混合采样\n(Ours)'], 
              metrics['Safety'], color=[color_old, color_new_b], width=0.4, edgecolor='black', alpha=0.85)
axs[1, 0].set_ylim(0, 7)
axs[1, 0].set_ylabel('最近邻极小间距 (m)', fontweight='bold', fontsize=12)
axs[1, 0].set_title('(C) 动力学极小安全防碰撞间距', fontsize=14, fontweight='bold')
axs[1, 0].axhline(safe_limit, color='red', linestyle='-.', linewidth=2, label=f'安全红线 ({safe_limit}m)')
axs[1, 0].fill_between([-0.5, 1.5], 0, safe_limit, color='red', alpha=0.1, hatch='//')
axs[1, 0].legend(loc='upper left')
axs[1, 0].text(0, metrics['Safety'][0]+0.15, f"{metrics['Safety'][0]:.2f}", ha='center', fontsize=13, fontweight='bold')
axs[1, 0].text(1, metrics['Safety'][1]+0.15, f"{metrics['Safety'][1]:.2f}", ha='center', color=color_new_b, fontsize=14, fontweight='bold')

# 先把 plt.subplots 生成的占位笛卡尔坐标轴删掉，避免重叠
axs[1, 1].remove()

#[1, 1] 多目标雷达图 (修复文字拥挤和图例阻挡)
ax4 = fig.add_subplot(2, 2, 4, polar=True)
categories =['特征细节辨识度', '算法时间效率', '安全防碰裕度', '结构抗干扰性', '几何模型保真度']
N_cat = len(categories)
scores_old =[4.0, 9.5, 2.0, 8.5, 6.0]
scores_new =[8.5, 8.0, 10.0, 9.0, 8.5]
scores_old += scores_old[:1]; scores_new += scores_new[:1]
angles =[n / float(N_cat) * 2 * pi for n in range(N_cat)]; angles += angles[:1]

ax4.plot(angles, scores_old, linewidth=2, color=color_old, label='传统体素基线')
ax4.fill(angles, scores_old, color=color_old, alpha=0.25)
ax4.plot(angles, scores_new, linewidth=2, color=color_new_r, label='特征感知优化 (Ours)')
ax4.fill(angles, scores_new, color=color_new_r, alpha=0.15)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=12, fontweight='bold')
# 【关键参数】pad=25：把文字标签向外推25个像素，完美避开图形
ax4.tick_params(axis='x', pad=22) 

ax4.set_ylim(0, 10)
ax4.set_yticks([2, 4, 6, 8, 10])
ax4.set_yticklabels(['2', '4', '6', '8', '10'], color="grey", size=9)

# 【关键参数】bbox_to_anchor 将图例移到最远处的独立空白角落，加个干净的白框
ax4.legend(loc='lower right', bbox_to_anchor=(1.35, -0.15), frameon=True, edgecolor='lightgray', facecolor='white', fontsize=11)
ax4.set_title('(D) 多目标协同寻优雷达图', fontsize=14, fontweight='bold', pad=30)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# bbox_inches='tight' 防止保存时外侧推出去的图例和标签被裁剪
plt.savefig('Clean_Ablation_Study_v2.png', dpi=300, bbox_inches='tight')
plt.show()
print("完美学术版四宫格已生成！请查看 Clean_Ablation_Study_v2.png")