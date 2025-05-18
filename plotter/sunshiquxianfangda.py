import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import ticker

plt.style.use(['science', 'nature'])
from matplotlib.backends.backend_pdf import PdfPages

# 设置颜色列表和反转颜色映射
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# 设置模型名称和MyMIT数据集
models = ['MLP', 'CNN', 'LSTM', 'KAN', 'KAN-PINN', 'Attention-KAN', 'Attention-KAN-PINN(without PDE)', 'Attention-KAN-PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# 为所有模型生成一个新的图
fig, ax = plt.subplots(figsize=(7, 5), dpi=600)

# 创建子图：在主图右下角放大显示小损失值区域
ax_inset = fig.add_axes([0.35, 0.20, 0.35, 0.35])  # 放大子图位置和大小
# 设置 y 轴的刻度数量减少一半
ax_inset.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower', nbins=3))
# 遍历模型名称并加载各自的损失数据
for model in models:
    try:
        # 设置文件路径
        root = f'../Final_Paper_results/{data}_{model} results1/Experiment10/'

        # 设置文件路径
        root = f'../Final_Paper_results/{data}_{model} results1/Experiment10/'

        # 假设损失值存储在epoch_losses.npy文件中，且每个epoch的损失是一个数组
        loss_file = root + 'epoch_losses.npy'
        loss = np.load(loss_file, allow_pickle=True)  # 假设文件是epoch_losses.npy
        print(f"Loaded loss data from {loss_file} for {model}.")

        # 打印损失数据的基本信息，调试用
        print(f"Total epochs: {len(loss)}")

        # 检查是否有数据
        if len(loss) == 0:
            raise ValueError(f"No loss data found in {loss_file}")

        epochs = np.arange(1, len(loss) + 1)  # 生成Epochs

        # 如果没有加载到损失数据，则跳过当前模型
        if len(loss) == 0:
            print(f"No loss data for {model}. Skipping...")
            continue

        # 过滤损失值，只保留小于等于0.020的损失
        filtered_epochs = epochs[loss <= 0.020]
        filtered_loss = loss[loss <= 0.020]

        # 绘制大图的损失曲线
        if model == 'MLP':
            ax.plot(filtered_epochs, filtered_loss, color='blue', linestyle='--', marker='s', markersize=2.5, alpha=0.7,
                    label=model)
        elif model == 'CNN':
            ax.plot(filtered_epochs, filtered_loss, color='green', linestyle='-.', marker='^', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'LSTM':
            ax.plot(filtered_epochs, filtered_loss, color='red', linestyle=':', marker='D', markersize=2.5, alpha=0.7,
                    label=model)
        elif model == 'KAN':
            ax.plot(filtered_epochs, filtered_loss, color='orange', linestyle='-', marker='x', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'KAN-PINN':
            ax.plot(filtered_epochs, filtered_loss, color='magenta', linestyle='-.', marker='p', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'Attention-KAN':
            ax.plot(filtered_epochs, filtered_loss, color='brown', linestyle='--', marker='*', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'Attention-KAN-PINN(without PDE)':
            ax.plot(filtered_epochs, filtered_loss, color='teal', linestyle=':', marker='h', markersize=2.5, alpha=0.7,
                    label=model)
        elif model == 'Attention-KAN-PINN':
            ax.plot(filtered_epochs, filtered_loss, color='purple', linestyle='-', marker='o', markersize=2.5,
                    alpha=0.7, label=model)

        # 过滤损失值小于0.001的部分
        small_loss_mask = loss <= 0.001
        small_epochs = epochs[small_loss_mask]
        small_loss = loss[small_loss_mask]

        # 在子图上绘制小损失值部分，使用与大图相同的颜色和样式
        if model == 'MLP':
            ax_inset.plot(small_epochs, small_loss, color='blue', linestyle='--', marker='s', markersize=2.5, alpha=0.7)
        elif model == 'CNN':
            ax_inset.plot(small_epochs, small_loss, color='green', linestyle='-.', marker='^', markersize=2.5,
                          alpha=0.7)
        elif model == 'LSTM':
            ax_inset.plot(small_epochs, small_loss, color='red', linestyle=':', marker='D', markersize=2.5, alpha=0.7)
        elif model == 'KAN':
            ax_inset.plot(small_epochs, small_loss, color='orange', linestyle='-', marker='x', markersize=2.5,
                          alpha=0.7)
        elif model == 'KAN-PINN':
            ax_inset.plot(small_epochs, small_loss, color='magenta', linestyle='-.', marker='p', markersize=2.5,
                          alpha=0.7)
        elif model == 'Attention-KAN':
            ax_inset.plot(small_epochs, small_loss, color='brown', linestyle='--', marker='*', markersize=2.5,
                          alpha=0.7)
        elif model == 'Attention-KAN-PINN(without PDE)':
            ax_inset.plot(small_epochs, small_loss, color='teal', linestyle=':', marker='h', markersize=2.5, alpha=0.7)
        elif model == 'Attention-KAN-PINN':
            ax_inset.plot(small_epochs, small_loss, color='purple', linestyle='-', marker='o', markersize=2.5,
                          alpha=0.7)

    except Exception as e:
        print(f"Error loading data for {model}: {e}")

# 设置大图的 x 轴和 y 轴标签
ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')  # 训练Epochs
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')  # 损失值

# 调整刻度值字体大小和加粗
ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout', grid_color='black',
               grid_alpha=0.5)  # 加粗刻度线和字体

# 添加网格
ax.grid(True, linestyle='--', alpha=0.6)

# 获取主图的图例句柄和标签
handles, labels = ax.get_legend_handles_labels()

# 将图例放在子图上方，排列为两列
ax_inset.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.7, 1.28),  # 控制图例在子图正上方位置
    ncol=2,                      # 分成两列显示
    fontsize=12,
    frameon=False
)





# 加粗图外框
plt.setp(ax.spines.values(), linewidth=1.5)

# 设置子图的网格
ax_inset.grid(True, linestyle='--', alpha=0.6)
ax_inset.tick_params(axis='both', which='major', labelsize=10, width=2, length=6, direction='inout', grid_color='black',
                     grid_alpha=0.5)
# 使用 `constrained_layout` 替代 `tight_layout`
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 手动调整间距，避免警告

# 保存损失曲线图像
plt.savefig('loss_curve_comparison_expanded_models.png', dpi=600, bbox_inches='tight')

# 展示图形
plt.show()