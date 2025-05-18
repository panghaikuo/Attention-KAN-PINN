'''
比较三个模型在MIT数据集上的结果'MAE', 'MAPE', 'MSE', 'RMSE', 'R2'
'''
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
KAN_MIT_results = pd.read_excel('../MIT results/MIT-KAN results.xlsx')
KAN_MIT_results1 = pd.read_excel('../MIT results/MIT-KAN results1.xlsx')
KAN_MIT_results2 = pd.read_excel('../MIT results/MIT-KAN results2.xlsx')
# 假设每个文件都有相同的列名
metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
colors = ['blue', 'orange', 'green']   # 不同文件的颜色

for metric in metrics:
    plt.figure(figsize=(10, 5))

    plt.plot(KAN_MIT_results['experiment'], KAN_MIT_results[metric], color=colors[0], label='KAN_MIT_results')
    plt.plot(KAN_MIT_results1['experiment'], KAN_MIT_results1[metric], color=colors[1], label='KAN_MIT_results1')
    plt.plot(KAN_MIT_results2['experiment'], KAN_MIT_results2[metric], color=colors[2], label='KAN_MIT_results2')

    plt.title(f'{metric} Comparison')
    plt.xlabel('experiment')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    # plt.savefig(f'{metric}_comparison.png')  # 保存图像
    plt.show()
