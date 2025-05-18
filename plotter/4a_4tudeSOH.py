import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots

plt.style.use(['science', 'nature'])

# 设置颜色列表和反转颜色映射
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# 设置模型列表和 MyMIT 数据集
# models = ['MLP', 'CNN', 'LSTM', 'Attention-KAN-PINN']
models = ['MLP', 'NEW', 'LSTM', 'Attention-KAN-PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# 创建整体图形和子图
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300, constrained_layout=True)

for i, model in enumerate(models):
    ax = axes[i]
    try:
        # 设置文件路径
        root = f'../Paper_results/{data}_{model} results/Experiment1/'
        pred_label = np.load(root + 'pred_label.npy')
        true_label = np.load(root + 'true_label.npy')
    except Exception as e:
        print(f"Error loading data for {model}: {e}")
        continue

    # 计算误差
    error = np.abs(pred_label - true_label)

    # 绘制散点图，使用反转的颜色映射
    sc = ax.scatter(true_label, pred_label, c=error, cmap=colors, s=10, alpha=0.7, vmin=0, vmax=0.1)
    ax.plot([0.65, 1.15], [0.65, 1.15], '--', c='#ff4d4e', alpha=1, linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlabel('True SOH', fontsize=14, fontweight='bold')  # 加粗并设置字体大小
    ax.set_ylabel('Prediction', fontsize=14, fontweight='bold') if i == 0 else ax.set_ylabel('')  # 仅第一个子图保留y轴标签

    # 设置 xlim 和 ylim
    ax.set_xlim(lims[data])
    ax.set_ylim(lims[data])

    # 减少 y 轴刻度数量
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # 设置最多显示 5 个刻度

    # 设置标题
    ax.set_title(f'{model}', fontsize=16, fontweight='bold')

    # 调整刻度值字体大小和加粗
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout', grid_color='black', grid_alpha=0.5)  # 加粗刻度线和字体

    # 加粗图外框
    plt.setp(ax.spines.values(), linewidth=1.5)

# 添加单独的颜色条到右边，并调整宽度和长度
cbar = fig.colorbar(
    sc, ax=axes, location='right', aspect=40, shrink=0.75, pad=0.08
)
cbar.set_label('Absolute error', fontsize=14, fontweight='bold')  # 设置颜色条标题加粗并放大
cbar.ax.tick_params(labelsize=12, width=1.5)  # 调整颜色条刻度加粗

# 保存图片
save_path = '4combined_figure.png'
plt.savefig(save_path, dpi=300, format='png')
print(f"Saved combined figure as {save_path}")

# 展示图形
plt.show()
