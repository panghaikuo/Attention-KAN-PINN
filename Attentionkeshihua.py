import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 自定义 WeightedAttentionLayer
class WeightedAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(WeightedAttentionLayer, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(input_dim))  # 可学习权重

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        attn_scores = torch.matmul(x, self.attn_weights)  # [batch_size, seq_len]
        attn_scores = torch.softmax(attn_scores, dim=1)  # 对序列维度归一化
        attn_scores = attn_scores.unsqueeze(-1)  # [batch_size, seq_len, 1]
        output = x * attn_scores  # 加权
        return output, attn_scores  # 返回注意力加权后结果和注意力权重

# 主模型结构
class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.attn = WeightedAttentionLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 注意力权重缓存
        self.attention_weights_cache = []

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        attn_output, attn_weights = self.attn(x)  # 获取注意力输出和权重
        self.attention_weights_cache.append(attn_weights.detach().cpu())  # 缓存注意力权重
        x = attn_output.mean(dim=1)  # 简化为平均池化，也可使用其他聚合方式
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output

# 可视化函数
def visualize_attention_weights(attn_weights_list, sample_idx, save_dir='attn_plots'):
    """
    attn_weights_list: List of attention weight tensors [batch_size, seq_len, 1]
    sample_idx: 第几个batch中第几个样本 (batch_id, sample_id)
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_id, sample_id = sample_idx
    if batch_id >= len(attn_weights_list):
        raise IndexError("batch_id超出范围")
    if sample_id >= attn_weights_list[batch_id].shape[0]:
        raise IndexError("sample_id超出范围")

    attn_weights = attn_weights_list[batch_id][sample_id].squeeze()  # [seq_len]
    plt.figure(figsize=(10, 2))
    sns.heatmap(attn_weights.unsqueeze(0).numpy(), cmap='viridis', cbar=True, annot=True, xticklabels=False)
    plt.title(f'Attention Weights (Batch {batch_id}, Sample {sample_id})')
    plt.xlabel('Feature Index')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'attn_batch{batch_id}_sample{sample_id}.png'))
    plt.close()

# 示例训练/推理流程（简化）
if __name__ == "__main__":
    batch_size = 32
    seq_len = 16  # 假设特征数为16
    input_dim = 1  # 每个位置一个特征值（也可以为多个维度）
    hidden_dim = 64
    output_dim = 1

    model = AttentionModel(input_dim, hidden_dim, output_dim)
    dummy_input = torch.rand(batch_size, seq_len, input_dim)

    # 模拟多个 batch 前向传播
    for _ in range(5):  # 假设有5个batch
        output = model(dummy_input)

    # 可视化第2个batch中第3个样本的注意力权重
    visualize_attention_weights(model.attention_weights_cache, sample_idx=(1, 2))
