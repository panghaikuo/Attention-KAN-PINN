import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
from scipy.stats import spearmanr


def analyze_sensitivity_results(base_folder='MIT_sensitivity_analysis'):
    """
    分析超参数敏感性实验的结果

    Args:
        base_folder: 包含所有敏感性实验的基础文件夹
    """
    # 创建结果目录
    results_folder = os.path.join(base_folder, 'analysis_results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # 分析单参数实验
    analyze_single_parameter(base_folder, 'lambda1', results_folder)
    analyze_single_parameter(base_folder, 'lambda2', results_folder)
    analyze_single_parameter(base_folder, 'lambda3', results_folder)
    analyze_single_parameter(base_folder, 'l2_lambda', results_folder)

    # 分析参数组合实验
    analyze_parameter_combination(base_folder, 'lambda1_lambda2', results_folder)
    analyze_parameter_combination(base_folder, 'lambda2_lambda3', results_folder)

    # 综合分析所有参数的敏感性
    analyze_overall_sensitivity(base_folder, results_folder)


def analyze_single_parameter(base_folder, param_name, results_folder):
    """
    分析单个参数的敏感性

    Args:
        base_folder: 基础实验文件夹
        param_name: 参数名称
        results_folder: 结果保存文件夹
    """
    print(f"分析 {param_name} 的敏感性...")

    # 参数文件夹路径
    param_folder = os.path.join(base_folder, f'{param_name}_sensitivity')
    if not os.path.exists(param_folder):
        print(f"警告: {param_folder} 不存在，跳过分析")
        return

    # 收集该参数所有实验结果
    experiments = []
    for exp_dir in os.listdir(param_folder):
        if not exp_dir.startswith(param_name):
            continue

        # 提取参数值
        param_value = float(exp_dir.split('_')[-1])

        # 读取实验摘要
        summary_path = os.path.join(param_folder, exp_dir, 'experiment_summary.txt')
        if not os.path.exists(summary_path):
            print(f"警告: {summary_path} 不存在，跳过")
            continue

        # 解析摘要文件
        best_metrics = {}
        with open(summary_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'MSE:' in line and 'Best metrics' in lines[lines.index(line) - 1]:
                    best_metrics['MSE'] = float(line.split(':')[-1].strip())
                elif 'MAE:' in line:
                    best_metrics['MAE'] = float(line.split(':')[-1].strip())
                elif 'MAPE:' in line:
                    best_metrics['MAPE'] = float(line.split(':')[-1].strip())
                elif 'RMSE:' in line:
                    best_metrics['RMSE'] = float(line.split(':')[-1].strip())
                elif 'Best epoch:' in line:
                    best_metrics['Best_epoch'] = int(line.split(':')[-1].strip())

        # 读取训练过程损失
        try:
            epoch_losses = np.load(os.path.join(param_folder, exp_dir, 'epoch_losses.npy'))
            valid_mses = np.load(os.path.join(param_folder, exp_dir, 'valid_mses.npy'))

            # 计算收敛速度 (找到验证MSE首次低于阈值的epoch)
            convergence_threshold = np.min(valid_mses) * 1.1  # 最小值的1.1倍作为阈值
            convergence_epoch = np.argmax(valid_mses <= convergence_threshold) + 1

            experiments.append({
                'param_value': param_value,
                'MSE': best_metrics.get('MSE', np.nan),
                'MAE': best_metrics.get('MAE', np.nan),
                'MAPE': best_metrics.get('MAPE', np.nan),
                'RMSE': best_metrics.get('RMSE', np.nan),
                'Best_epoch': best_metrics.get('Best_epoch', np.nan),
                'convergence_epoch': convergence_epoch,
                'min_valid_mse': np.min(valid_mses),
                'final_loss': epoch_losses[-1] if len(epoch_losses) > 0 else np.nan,
                'epoch_losses': epoch_losses,
                'valid_mses': valid_mses
            })
        except:
            print(f"警告: 无法加载损失数据 {exp_dir}")

    if not experiments:
        print(f"未找到有效的 {param_name} 敏感性实验数据")
        return

    # 转换为DataFrame并按参数值排序
    df = pd.DataFrame(experiments)
    df = df.sort_values('param_value')

    # 保存为CSV
    csv_path = os.path.join(results_folder, f'{param_name}_sensitivity.csv')
    df[['param_value', 'MSE', 'MAE', 'MAPE', 'RMSE', 'Best_epoch', 'convergence_epoch', 'min_valid_mse',
        'final_loss']].to_csv(csv_path, index=False)

    # 计算Spearman相关系数，分析参数与性能指标的关系
    correlations = {}
    metrics = ['MSE', 'MAE', 'MAPE', 'RMSE', 'convergence_epoch']
    for metric in metrics:
        if all(~np.isnan(df[metric])):
            corr, p_value = spearmanr(df['param_value'], df[metric])
            correlations[metric] = (corr, p_value)

    # 计算敏感度得分 (各指标相关系数的绝对值平均)
    sensitivity_score = np.mean([abs(corr) for corr, _ in correlations.values()])

    # 绘制性能指标变化图
    plt.figure(figsize=(15, 10))

    # MSE随参数变化
    plt.subplot(2, 2, 1)
    plt.plot(df['param_value'], df['MSE'], 'o-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'MSE vs {param_name}', fontsize=14)
    plt.grid(True)

    # MAE随参数变化
    plt.subplot(2, 2, 2)
    plt.plot(df['param_value'], df['MAE'], 'o-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(f'MAE vs {param_name}', fontsize=14)
    plt.grid(True)

    # 收敛速度随参数变化
    plt.subplot(2, 2, 3)
    plt.plot(df['param_value'], df['convergence_epoch'], 'o-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Convergence Epoch', fontsize=12)
    plt.title(f'Convergence Speed vs {param_name}', fontsize=14)
    plt.grid(True)

    # 训练损失曲线比较
    plt.subplot(2, 2, 4)
    for i, row in df.iterrows():
        plt.plot(row['epoch_losses'], label=f"{param_name}={row['param_value']}")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'{param_name}_sensitivity_analysis.png'), dpi=300)
    plt.close()

    # 保存敏感度分析结果
    with open(os.path.join(results_folder, f'{param_name}_sensitivity_analysis.txt'), 'w') as f:
        f.write(f"敏感度分析结果 - {param_name}\n")
        f.write(f"==============================\n")
        f.write(f"敏感度得分: {sensitivity_score:.4f}\n\n")
        f.write("相关系数 (Spearman):\n")
        for metric, (corr, p_value) in correlations.items():
            f.write(f"  {metric}: {corr:.4f} (p={p_value:.4f})\n")

        f.write("\n最优参数值:\n")
        best_idx = df['MSE'].idxmin()
        f.write(f"  基于MSE: {param_name}={df.loc[best_idx, 'param_value']:.6f} (MSE={df.loc[best_idx, 'MSE']:.6f})\n")

        best_idx = df['MAE'].idxmin()
        f.write(f"  基于MAE: {param_name}={df.loc[best_idx, 'param_value']:.6f} (MAE={df.loc[best_idx, 'MAE']:.6f})\n")

        # 找出敏感区域 (相邻点之间性能变化最大的区域)
        if len(df) >= 2:
            mse_changes = np.abs(np.diff(df['MSE'].values) / np.diff(df['param_value'].values))
            if len(mse_changes) > 0:
                max_change_idx = np.argmax(mse_changes)
                f.write("\n敏感区域:\n")
                f.write(
                    f"  MSE变化率最大的区间: {param_name} ∈ [{df['param_value'].iloc[max_change_idx]:.6f}, {df['param_value'].iloc[max_change_idx + 1]:.6f}]\n")
                f.write(f"  该区间MSE变化率: {mse_changes[max_change_idx]:.6f} per unit\n")

    print(f"{param_name} 敏感性分析完成。")
    return df


def analyze_parameter_combination(base_folder, combo_name, results_folder):
    """
    分析参数组合的敏感性

    Args:
        base_folder: 基础实验文件夹
        combo_name: 参数组合名称 (如 'lambda1_lambda2')
        results_folder: 结果保存文件夹
    """
    print(f"分析 {combo_name} 组合敏感性...")

    param_names = combo_name.split('_')
    if len(param_names) != 2:
        print(f"警告: {combo_name} 不是有效的参数组合名称")
        return

    param1, param2 = param_names

    # 参数组合文件夹
    combo_folder = os.path.join(base_folder, f'{combo_name}_sensitivity')
    if not os.path.exists(combo_folder):
        print(f"警告: {combo_folder} 不存在，跳过分析")
        return

    # 收集组合实验结果
    experiments = []
    pattern = re.compile(f"{param1}_(.+)_{param2}_(.+)")

    for exp_dir in os.listdir(combo_folder):
        match = pattern.match(exp_dir)
        if not match:
            continue

        param1_value = float(match.group(1))
        param2_value = float(match.group(2))

        # 读取实验摘要
        summary_path = os.path.join(combo_folder, exp_dir, 'experiment_summary.txt')
        if not os.path.exists(summary_path):
            print(f"警告: {summary_path} 不存在，跳过")
            continue

        # 解析摘要文件
        best_metrics = {}
        with open(summary_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'MSE:' in line and 'Best metrics' in lines[lines.index(line) - 1]:
                    best_metrics['MSE'] = float(line.split(':')[-1].strip())
                elif 'MAE:' in line:
                    best_metrics['MAE'] = float(line.split(':')[-1].strip())

        experiments.append({
            'param1_value': param1_value,
            'param2_value': param2_value,
            'MSE': best_metrics.get('MSE', np.nan),
            'MAE': best_metrics.get('MAE', np.nan)
        })

    if not experiments:
        print(f"未找到有效的 {combo_name} 敏感性实验数据")
        return

    # 转换为DataFrame
    df = pd.DataFrame(experiments)

    # 保存为CSV
    csv_path = os.path.join(results_folder, f'{combo_name}_sensitivity.csv')
    df.to_csv(csv_path, index=False)

    # 创建MSE的热力图
    try:
        # 获取参数1和参数2的唯一值
        param1_values = sorted(df['param1_value'].unique())
        param2_values = sorted(df['param2_value'].unique())

        # 创建网格
        mse_grid = np.full((len(param1_values), len(param2_values)), np.nan)
        mae_grid = np.full((len(param1_values), len(param2_values)), np.nan)

        # 填充网格
        for i, p1 in enumerate(param1_values):
            for j, p2 in enumerate(param2_values):
                mask = (df['param1_value'] == p1) & (df['param2_value'] == p2)
                if mask.any():
                    mse_grid[i, j] = df.loc[mask, 'MSE'].values[0]
                    mae_grid[i, j] = df.loc[mask, 'MAE'].values[0]

        # 绘制MSE热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(mse_grid, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='MSE')
        plt.xticks(np.arange(len(param2_values)), param2_values)
        plt.yticks(np.arange(len(param1_values)), param1_values)
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.title(f'MSE Heatmap for {param1} vs {param2}')
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(mse_grid[i, j]):
                    plt.text(j, i, f"{mse_grid[i, j]:.4f}",
                             ha="center", va="center",
                             color="white" if mse_grid[i, j] > np.nanmean(mse_grid) else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{combo_name}_mse_heatmap.png'), dpi=300)
        plt.close()

        # 找出最优组合
        min_idx = np.nanargmin(mse_grid)
        min_i, min_j = np.unravel_index(min_idx, mse_grid.shape)
        optimal_param1 = param1_values[min_i]
        optimal_param2 = param2_values[min_j]

        # 保存分析结果
        with open(os.path.join(results_folder, f'{combo_name}_analysis.txt'), 'w') as f:
            f.write(f"参数组合分析结果 - {combo_name}\n")
            f.write(f"==============================\n")
            f.write(f"最优组合 (基于MSE):\n")
            f.write(f"  {param1} = {optimal_param1}\n")
            f.write(f"  {param2} = {optimal_param2}\n")
            f.write(f"  MSE = {mse_grid[min_i, min_j]:.6f}\n\n")

            f.write(f"参数相互作用分析:\n")
            # 计算参数交互作用
            row_effects = np.nanmean(mse_grid, axis=1) - np.nanmean(mse_grid)
            col_effects = np.nanmean(mse_grid, axis=0) - np.nanmean(mse_grid)

            interaction = mse_grid - (
                        np.nanmean(mse_grid) + np.reshape(row_effects, (-1, 1)) + np.reshape(col_effects, (1, -1)))

            # 计算交互强度
            interaction_strength = np.nanstd(interaction) / np.nanstd(mse_grid)

            f.write(f"  交互强度: {interaction_strength:.4f}\n")
            if interaction_strength > 0.3:
                f.write(f"  结论: 存在强交互作用，参数不应独立优化\n")
            elif interaction_strength > 0.1:
                f.write(f"  结论: 存在中等交互作用，参数优化应考虑组合效应\n")
            else:
                f.write(f"  结论: 交互作用较弱，参数可以相对独立优化\n")

    except Exception as e:
        print(f"绘制热力图时出错: {e}")

    print(f"{combo_name} 组合敏感性分析完成。")
    return df


def analyze_overall_sensitivity(base_folder, results_folder):
    """
    综合分析所有参数的敏感性

    Args:
        base_folder: 基础实验文件夹
        results_folder: 结果保存文件夹
    """
    print("进行整体敏感性分析...")

    # 读取各参数的敏感性分析结果
    param_names = ['lambda1', 'lambda2', 'lambda3', 'l2_lambda']
    sensitivity_scores = {}

    for param in param_names:
        analysis_file = os.path.join(results_folder, f'{param}_sensitivity_analysis.txt')
        if not os.path.exists(analysis_file):
            print(f"警告: {analysis_file} 不存在，跳过")
            continue

        with open(analysis_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '敏感度得分:' in line:
                    sensitivity_scores[param] = float(line.split(':')[-1].strip())
                    break

    if not sensitivity_scores:
        print("没有找到足够的敏感性分析结果来进行综合分析")
        return

    # 创建敏感度比较图
    plt.figure(figsize=(10, 6))
    params = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[p] for p in params]

    plt.bar(params, scores, color='steelblue')
    plt.ylabel('Sensitivity Score', fontsize=12)
    plt.title('Parameter Sensitivity Comparison', fontsize=14)
    plt.xticks(rotation=0)

    # 添加数值标签
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'overall_sensitivity_comparison.png'), dpi=300)
    plt.close()

    # 保存综合分析结果
    with open(os.path.join(results_folder, 'overall_sensitivity_analysis.txt'), 'w') as f:
        f.write("超参数敏感性综合分析\n")
        f.write("============================\n\n")

        # 按敏感度排序参数
        sorted_params = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)

        f.write("参数敏感度排名 (从高到低):\n")
        for param, score in sorted_params:
            f.write(f"  {param}: {score:.4f}\n")

        f.write("\n调优建议:\n")
        f.write("1. 首先调整敏感度最高的参数，因为它们对模型性能影响最大\n")
        most_sensitive = sorted_params[0][0] if sorted_params else ""
        f.write(f"2. {most_sensitive} 是最敏感的参数，在调优时应该优先考虑\n")
        f.write("3. 参数组合分析表明某些参数之间可能存在交互作用，建议结合组合分析结果进行调优\n")
        f.write("4. 对于低敏感度的参数，可以使用默认值或在较粗的网格上搜索\n")

    print("整体敏感性分析完成。")


def plot_convergence_comparison(base_folder, results_folder):
    """
    比较不同参数设置下的收敛曲线

    Args:
        base_folder: 基础实验文件夹
        results_folder: 结果保存文件夹
    """
    print("绘制收敛曲线比较...")

    # 为每个参数找出最佳设置
    param_names = ['lambda1', 'lambda2', 'lambda3', 'l2_lambda']
    best_settings = {}

    for param in param_names:
        csv_path = os.path.join(results_folder, f'{param}_sensitivity.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if 'MSE' in df.columns and not df['MSE'].isna().all():
            best_idx = df['MSE'].idxmin()
            best_value = df.loc[best_idx, 'param_value']
            best_settings[param] = best_value

    # 找出各最佳设置对应的验证MSE曲线
    best_valid_curves = {}
    for param, value in best_settings.items():
        param_folder = os.path.join(base_folder, f'{param}_sensitivity')
        if not os.path.exists(param_folder):
            continue

        exp_dir = f"{param}_{value}"
        # 查找匹配的文件夹
        matched_dirs = [d for d in os.listdir(param_folder) if d.startswith(exp_dir)]
        if not matched_dirs:
            continue

        valid_mses_path = os.path.join(param_folder, matched_dirs[0], 'valid_mses.npy')
        if os.path.exists(valid_mses_path):
            valid_mses = np.load(valid_mses_path)
            best_valid_curves[f"{param}={value}"] = valid_mses

    # 绘制收敛曲线比较
    if best_valid_curves:
        plt.figure(figsize=(12, 6))
        for label, curve in best_valid_curves.items():
            epochs = np.arange(1, len(curve) + 1)
            plt.plot(epochs, curve, label=label, linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation MSE', fontsize=12)
        plt.title('Convergence Curves for Best Parameter Settings', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 使用对数刻度更容易比较收敛速度
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'best_convergence_comparison.png'), dpi=300)
        plt.close()

    print("收敛曲线比较完成。")


def create_optimal_config(base_folder, results_folder):
    """
    基于敏感性分析创建最优配置文件

    Args:
        base_folder: 基础实验文件夹
        results_folder: 结果保存文件夹
    """
    print("创建最优配置文件...")

    # 获取每个参数的最优值
    optimal_params = {}
    param_names = ['lambda1', 'lambda2', 'lambda3', 'l2_lambda']

    for param in param_names:
        csv_path = os.path.join(results_folder, f'{param}_sensitivity.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if 'MSE' in df.columns and not df['MSE'].isna().all():
            best_idx = df['MSE'].idxmin()
            optimal_params[param] = df.loc[best_idx, 'param_value']

    # 考虑组合分析结果
    combo_names = ['lambda1_lambda2', 'lambda2_lambda3']
    for combo in combo_names:
        analysis_file = os.path.join(results_folder, f'{combo}_analysis.txt')
        if not os.path.exists(analysis_file):
            continue

        param1, param2 = combo.split('_')
        with open(analysis_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '最优组合' in line:
                    for j in range(1, 3):
                        if i + j < len(lines) and param1 in lines[i + j]:
                            optimal_params[param1] = float(lines[i + j].split('=')[-1].strip())
                        if i + j < len(lines) and param2 in lines[i + j]:
                            optimal_params[param2] = float(lines[i + j].split('=')[-1].strip())

    # 创建最优配置文件
    with open(os.path.join(results_folder, 'optimal_configuration.txt'), 'w') as f:
        f.write("超参数敏感性分析最优配置\n")
        f.write("============================\n\n")
        f.write("推荐设置:\n")
        for param, value in optimal_params.items():
            f.write(f"{param} = {value}\n")

    print("最优配置文件创建完成。")


if __name__ == "__main__":
    # 设置基础目录，根据实际情况修改
    base_folder = 'MIT_sensitivity_analysis1'

    # 分析敏感性数据
    analyze_sensitivity_results(base_folder)

    # 绘制收敛曲线比较
    results_folder = os.path.join(base_folder, 'analysis_results')
    plot_convergence_comparison(base_folder, results_folder)

    # 创建最优配置推荐
    create_optimal_config(base_folder, results_folder)

    print("超参数敏感性分析完成!")