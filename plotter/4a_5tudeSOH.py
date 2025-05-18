import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots

plt.style.use(['science', 'nature'])

# 设置颜色列表
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# 扩展模型列表
models = ['KAN', 'KAN-PINN', 'Attention-KAN', 'Attention-KAN-PINN(without PDE)', 'Attention-KAN-PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# 创建子图（5个）
fig, axes = plt.subplots(1, len(models), figsize=(20, 5), dpi=300, constrained_layout=True)

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

    # 计算绝对误差
    error = np.abs(pred_label - true_label)

    # 绘制散点图
    sc = ax.scatter(true_label, pred_label, c=error, cmap=colors, s=10, alpha=0.7, vmin=0, vmax=0.1)
    ax.plot([0.65, 1.15], [0.65, 1.15], '--', c='#ff4d4e', alpha=1, linewidth=1)
    ax.set_aspect('equal')

    # 标签设置
    ax.set_xlabel('True SOH', fontsize=14, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Prediction', fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.set_xlim(lims[data])
    ax.set_ylim(lims[data])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(f'{model}', fontsize=15, fontweight='bold')

    # 刻度样式
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout')

    # 加粗图框
    plt.setp(ax.spines.values(), linewidth=1.5)

# 添加统一颜色条
cbar = fig.colorbar(sc, ax=axes, location='right', aspect=40, shrink=0.75, pad=0.08)
cbar.set_label('Absolute error', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12, width=1.5)

# 保存图像
save_path = 'combined_figure_with_KANPINN.png'
plt.savefig(save_path, dpi=300, format='png')
print(f"Saved combined figure as {save_path}")

plt.show()
