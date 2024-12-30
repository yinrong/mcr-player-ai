import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})  # 14 是字体大小，可以改为你需要的值


# 如果你本地安装了中文字体，可在此设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 6))

# 关闭坐标刻度，设置绘图范围
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 4.5)

# ========== 绘制网格线 (6列×4行) ==========
line_kwargs = dict(color='black', linewidth=1.0, linestyle='--')
# 竖线 x=0..6
for x in range(6):
    ax.plot([x, x], [0, 4], **line_kwargs)
# 横线 y=0..4
for y in range(4):
    ax.plot([0, 6], [y, y], **line_kwargs)

# ========== 箭头 1：垂直，从 (0,0) 到 (0,4)，文字在上端 ==========
# 这里使用 arrowstyle='<-'，表示箭头头部在 xytext
# xy 指注释点(箭尾)，xytext 指文字及箭头头部所在点
ax.annotate(
    "3C制造企业运营需求",
    xy=(0, 0),           # 箭尾
    xytext=(0, 4),       # 箭头末端 & 文本所在点
    xycoords='data',
    textcoords='data',
    arrowprops=dict(arrowstyle="<-", lw=1.5),
    
    ha='center',         # 水平居中
    va='bottom',         # 文字基线贴在 (0,4) 之下
    #rotation=90          # 让文字竖排
)

# ========== 箭头 2：水平，从 (0,0) 到 (6,0)，文字在右端 ==========
ax.annotate(
    "生产全流程",
    xy=(0, 0),            # 箭尾
    xytext=(6, -0.07),        # 箭头末端 & 文本所在点
    xycoords='data',
    textcoords='data',
    arrowprops=dict(arrowstyle="<-", lw=1.5),
    
    ha='left',            # 让文字从 (6,0) 的左边开始
    va='bottom'
)

# ========== 左侧行标签(Q / D / C / F) ==========
y_labels = ["高制造良率(Q)", "高制造效率(D)", "低制造成本(C)", "高制造柔性(F)"]
for i, label in enumerate(y_labels):
    # 从上到下分别是3,2,1,0，此处在网格中间稍靠左
    y_pos = 3 - i + 0.5
    ax.text(-0.2, y_pos, label, va='center', ha='right', )

# ========== 下方列标签 ==========
x_labels = ["工厂建设", "计划调度", "生产作业", "仓储物流", "设备管理", "能源管理"]
for j, label in enumerate(x_labels):
    x_pos = j + 0.5
    ax.text(x_pos, -0.2, label, ha='center', va='top', )

# ========== 设置文字方块的样式 (方形浅黄底, 黑边框) ==========
bbox_props = dict(boxstyle="square,pad=0.5", fc="#FFF2AE", ec="black")

# 下面在各格子里放置文字，根据需要稍微上下微调 y 值，让它离网格线更近
ax.text(2.5, 3.7, "工艺优化", ha='center', va='center',
        bbox=bbox_props)
ax.text(2.5, 3.3, "先进过程\n控制", ha='center', va='center',
        bbox=bbox_props)
ax.text(4.5, 3.7, "设备在线\n监测", ha='center', va='center',
        bbox=bbox_props)

ax.text(0.5, 2.6, "数字孪生", ha='center', va='center',
        bbox=bbox_props)
ax.text(1.5, 2.6, "资源动态\n配置", ha='center', va='center',
        bbox=bbox_props)
ax.text(2.5, 2.6, "智能协同\n作业", ha='center', va='center',
        bbox=bbox_props)
ax.text(4.5, 2.6, "设备运行\n优化", ha='center', va='center',
        bbox=bbox_props)

ax.text(0.5, 1.4, "工厂数字\n化建设", ha='center', va='center',
        bbox=bbox_props)
ax.text(1.5, 1.4, "计划优化", ha='center', va='center',
        bbox=bbox_props)
ax.text(3.5, 1.4, "精准配送", ha='center', va='center',
        bbox=bbox_props)
ax.text(4.5, 1.4, "设备故障\n诊断", ha='center', va='center',
        bbox=bbox_props)
ax.text(5.5, 1.4, "能耗平衡\n与优化", ha='center', va='center',
        bbox=bbox_props)

ax.text(1.5, 0.6, "智能排产", ha='center', va='center',
        bbox=bbox_props)
ax.text(3.5, 0.6, "智能仓储", ha='center', va='center',
        bbox=bbox_props)

# 去除图像四周边框
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
