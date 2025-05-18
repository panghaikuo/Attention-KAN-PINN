import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
import random
# plt.style.use(['science','nature'])
# 设置文件路径和特征映射
data_path = '../Dataset_plot/My data/MIT data/2017-05-12'
feature_map = {
    'CT': 'F1', 'CC_time': 'F2', 'CV_time': 'F3', 'CV_Q': 'F4',
    'voltage_slope': 'F5', 'voltage_entropy': 'F6', 'voltage_kurtosis': 'F7',
    'voltage_skewness': 'F8', 'CC_Q': 'F9', 'current_slope': 'F10',
    'current_entropy': 'F11', 'TM': 'F12', 'TX': 'F13', 'TA': 'F14',
    'IR': 'F15', 'QC': 'F16'
}

# 获取 ID 为 5 的倍数的电池文件
battery_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
battery_files = [f for i, f in enumerate(battery_files) if (i + 1) % 5 == 0]

# 分配颜色
colors = plt.cm.tab20.colors  # 使用 matplotlib 的 tab20 颜色图
battery_colors = {file: random.choice(colors) for file in battery_files}

# 创建归一化器
scaler = MinMaxScaler(feature_range=(-1, 1))

# 创建 4x4 的图形方框
fig, axs = plt.subplots(4, 4, figsize=(15, 15),dpi=300)  # 设置整体图形大小

# 设置全局字体和图表外观
plt.rcParams.update({
    'font.family': 'Arial',  # 设置字体为 Arial
    'font.size': 14,         # 设置字体大小
    'axes.labelsize': 14,    # 坐标轴标签字体大小
    'axes.titlesize': 16,    # 子图标题字体大小
    'legend.fontsize': 12,   # 图例字体大小
    'lines.antialiased': True,  # 启用抗锯齿
    'figure.dpi': 300,  # 提高图形分辨率
    'savefig.dpi': 300,  # 提高保存图形的分辨率
})

# 绘制每个特征的散点图
# 绘制每个特征的散点图
# 修改代码：将x轴表示归一化值，y轴表示SOH
for idx, (feature, feature_id) in enumerate(feature_map.items()):
    row, col = divmod(idx, 4)  # 计算子图在 4x4 网格中的位置
    ax = axs[row, col]

    # 遍历每个电池文件并绘制对应特征的散点图
    for file in battery_files:
        battery_data = pd.read_csv(os.path.join(data_path, file))

        # 提取 SOH 和当前特征
        soh = battery_data['capacity']
        feature_data = battery_data[[feature]]

        # 对特征进行 [-1, 1] 归一化
        feature_normalized = scaler.fit_transform(feature_data)

        # 绘制散点图（将x轴换为归一化的值，y轴为SOH）
        ax.scatter(feature_normalized, soh, color=battery_colors[file],
                   label=f'Battery ID: {file.split(".")[0]}', alpha=0.7, s=10)

    # 设置子图标题（上方显示特征名称，如CT、CC_time）
    ax.set_title(f'{feature}', fontsize=16, weight='bold')

    # 设置 x 轴刻度只显示 -1, 0, 1
    ax.set_xticks([-1, 0, 1])

    # 在每个子图的右侧显示特征编号（F1, F2, ..., F16），字体加大并加粗
    ax.text(1.05, 0.5, f'{feature_id}', transform=ax.transAxes, rotation=90, va='center', ha='left',
            fontsize=16, weight='bold', color='black')

    # 设置底部子图的 x 轴标签
    if row == 3:
        ax.set_xlabel('Normalized Value', fontsize=14, weight='bold')
    else:
        ax.set_xticklabels([])  # 其他行不显示 x 轴刻度

    # 设置左侧子图的 y 轴标签
    if col == 0:
        ax.set_ylabel('SOH', fontsize=14, weight='bold')
    else:
        ax.set_yticklabels([])  # 其他列不显示 y 轴刻度

    # 设置刻度字体大小
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴刻度字体大小
    ax.tick_params(axis='y', labelsize=16)  # 设置y轴刻度字体大小

    # 加粗坐标轴框线
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 加粗坐标轴的边框线条


# 创建颜色标签（颜色圆点在左，ID 在右）
color_patches = [Patch(color=color, label=file.split('.')[0]) for file, color in battery_colors.items()]

# 调整底部图例的位置，确保图例正常显示在下方
plt.legend(handles=color_patches,
           bbox_to_anchor=(1.4, -0.3),  # 向右移动图例
           ncol=5,  # 设定图例列数
           fontsize=12,  # 增大字体大小
           labelspacing=1.5,  # 增加图例项之间的间距
           handletextpad=2.5,  # 增加文本与图例符号之间的间距
           frameon=True,  # 显示图例的边框
           fancybox=True,  # 使用圆角边框
           borderpad=1,  # 调整图例边框与内容的间距
           title_fontsize=14,  # 设置图例标题字体大小
           prop={'weight': 'bold'})  # 图例字体加粗
# 调整布局，防止重叠
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 调整底部空间，确保图例显示完整
plt.savefig('guiyihua.png')
plt.show()

