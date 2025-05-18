import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 文件路径和其他参数设置
root = '../Dataset_plot/My data/MIT data'
folders = ['2017-05-12', '2017-06-30', '2018-04-12']
colors = ['#80A6E2',
'#7BDFF2',
'#FBDD85']
markers = ['o', 'v', 'D']
legends = ['Batch 2017-05-12', 'Batch 2017-06-30', 'Batch 2018-04-12']
line_width = 1.0
window_size = 5  # 平滑窗口大小

# 开始绘图
fig = plt.figure(figsize=(6, 3), dpi=600)

for i, folder in enumerate(folders):
    folder_path = os.path.join(root, folder)
    files = os.listdir(folder_path)

    for f in files:
        file_path = os.path.join(folder_path, f)
        data = pd.read_csv(file_path)

        # 提取容量数据并过滤掉超过1.1的值
        capacity = data['capacity'].values
        capacity = [c if c <= 1.1 else None for c in capacity]  # 超过1.1的值用None替代
        capacity_smoothed = pd.Series(capacity).fillna(method='ffill').rolling(window=window_size,
                                                                               min_periods=1).mean()  # 平滑容量曲线

        # 绘制平滑后的曲线
        plt.plot(capacity_smoothed, color=colors[i], alpha=1, linewidth=line_width,
                 marker=markers[i], markersize=2, markevery=50)

# 设置图例和标签
plt.xlabel('Cycle', fontsize=12, fontweight='bold')
plt.ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
custom_lines = [
    Line2D([0], [0], color=colors[0], linewidth=line_width, marker=markers[0], markersize=3.0),
    Line2D([0], [0], color=colors[1], linewidth=line_width, marker=markers[1], markersize=3.0),
    Line2D([0], [0], color=colors[2], linewidth=line_width, marker=markers[2], markersize=3.0)
]
plt.legend(custom_lines, legends, loc='upper right', frameon=False, fontsize=9)

plt.tick_params(axis='both', labelsize=12, width=2)  # 增加刻度线宽和加粗刻度值

# 加粗图框
ax = plt.gca()  # 获取当前坐标轴
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# 设置y轴范围和显示图表
plt.ylim([0.8, 1.2])
plt.tight_layout()
plt.savefig('MIT_cycle.png', dpi=600)
plt.show()
