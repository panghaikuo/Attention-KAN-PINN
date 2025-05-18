import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter, get_logger, eval_metrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'

class WeightedAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, layer_id=0):
        super(WeightedAttentionLayer, self).__init__()
        # 权重是一个可学习的参数，表示每个输入特征的注意力权重
        self.attn_weight = nn.Parameter(torch.randn(hidden_dim))  # (hidden_dim,) - 每个特征一个权重
        self.layer_id = layer_id  # 记录层的ID，用于保存权重
        self.feature_before = None  # 存储注意力应用前的特征
        self.feature_after = None  # 存储注意力应用后的特征
        self.current_weights = None  # 存储当前的权重值

    def forward(self, x):
        # 存储注意力应用前的特征
        self.feature_before = x.detach().cpu().mean(dim=0).numpy()  # 取批次平均

        # 计算每个特征的注意力权重
        # attn_weight: (hidden_dim,) 和 x: (batch_size, seq_len, hidden_dim) 通过广播相乘
        weight = torch.sigmoid(self.attn_weight)  # 使用sigmoid将权重约束在0到1之间
        self.current_weights = weight.detach().cpu().numpy()  # 存储当前权重

        weighted_x = x * weight  # 对每个特征进行加权

        # 存储注意力应用后的特征
        self.feature_after = weighted_x.detach().cpu().mean(dim=0).numpy()  # 取批次平均

        return weighted_x

    def save_attention_info(self, save_dir):
        """保存注意力机制的信息到文件"""
        os.makedirs(save_dir, exist_ok=True)

        if self.feature_before is not None and self.feature_after is not None and self.current_weights is not None:
            # 保存数据
            data = {
                'feature_before': self.feature_before,
                'feature_after': self.feature_after,
                'attention_weights': self.current_weights
            }
            np.save(os.path.join(save_dir, f'attention_layer_{self.layer_id}_data.npy'), data)

            # 可视化并保存图像
            self._visualize_attention(save_dir)

    def _visualize_attention(self, save_dir):
        """可视化注意力权重及其影响"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # 绘制应用注意力前的特征
        sns.heatmap(self.feature_before.reshape(1, -1), ax=axes[0], cmap='viridis', annot=False)
        axes[0].set_title(f'Layer {self.layer_id}: Features Before Attention')
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Value')

        # 绘制注意力权重
        sns.heatmap(self.current_weights.reshape(1, -1), ax=axes[1], cmap='Reds', annot=False)
        axes[1].set_title(f'Layer {self.layer_id}: Attention Weights')
        axes[1].set_xlabel('Feature Dimension')
        axes[1].set_ylabel('Weight Value')

        # 绘制应用注意力后的特征
        sns.heatmap(self.feature_after.reshape(1, -1), ax=axes[2], cmap='viridis', annot=False)
        axes[2].set_title(f'Layer {self.layer_id}: Features After Attention')
        axes[2].set_xlabel('Feature Dimension')
        axes[2].set_ylabel('Value')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'attention_layer_{self.layer_id}_visualization.png'))
        plt.close()

        # 绘制注意力前后特征变化的比较图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.feature_before))
        plt.plot(x, self.feature_before, label='Before Attention', alpha=0.7)
        plt.plot(x, self.feature_after, label='After Attention', alpha=0.7)
        plt.bar(x, self.current_weights, alpha=0.3, label='Attention Weights', color='red')
        plt.title(f'Layer {self.layer_id}: Feature Changes Before and After Attention')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, f'attention_layer_{self.layer_id}_comparison.png'))
        plt.close()


class KAN(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, hidden_dim=60, num_hidden_layers=4, dropout=0.0):
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention_layers = []  # 存储所有的注意力层，方便后续收集数据

        # 增加多层隐藏层和非线性激活函数
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        # 在每个隐藏层后添加加权注意力层
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            attention_layer = WeightedAttentionLayer(hidden_dim, layer_id=i)  # 加权注意力层
            layers.append(attention_layer)
            self.attention_layers.append(attention_layer)  # 添加到列表中

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

    def save_attention_data(self, save_dir):
        """保存所有注意力层的数据"""
        for layer in self.attention_layers:
            layer.save_attention_info(save_dir)


class Predictor(nn.Module):
    def __init__(self, input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = KAN(input_dim=17, output_dim=32, hidden_dim=60, num_hidden_layers=4, dropout=0.0)
        self.predictor = Predictor(input_dim=32)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def get_embedding(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def save_attention_data(self, save_dir):
        """保存编码器中所有注意力层的数据"""
        self.encoder.save_attention_data(save_dir)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


class LR_Scheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0

    def step(self):
        lr = self.lr_schedule[self.iter]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.iter += 1
        return lr

    def get_lr(self):
        return self.lr_schedule[self.iter]


class PINN(nn.Module):
    def __init__(self, args):
        super(PINN, self).__init__()
        self.args = args

        if args.save_folder and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u().to(device)
        self.dynamical_F = KAN(input_dim=35, output_dim=1, hidden_dim=args.F_hidden_dim, num_hidden_layers=3,
                               dropout=0.0).to(device)

        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 新增损失比例参数
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.l2_lambda = args.l2_lambda  # 你可以根据需要调整默认值
        # self.alpha = self.args.alpha
        # self.beta = self.args.beta
        self.best_model = None

        # 创建注意力权重保存目录
        self.attention_dir = os.path.join(args.save_folder, 'attention_data') if args.save_folder else 'attention_data'
        os.makedirs(self.attention_dir, exist_ok=True)

    def compute_l2_regularization(self):
        l2_reg = 0
        # 计算所有参数的平方和
        for param in self.solution_u.parameters():
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        for param in self.dynamical_F.parameters():
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        return self.l2_lambda * l2_reg

    def _save_args(self):
        if self.args.log_dir:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.info(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self, xt):
        return self.solution_u(xt)

    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)

        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()

    def forward(self, xt):
        xt = xt.detach().requires_grad_(True)
        x, t = xt[:, :-1], xt[:, -1:]
        u = self.solution_u(torch.cat((x, t), dim=1))
        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True)[0]
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        return u, u_t - F

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter, loss2_meter, loss3_meter, l2_reg_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for idx, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            # 使用lambda1, lambda2, lambda3控制不同损失项的比例
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
            loss2 = 0.5 * self.loss_func(f1, torch.zeros_like(f1)) + 0.5 * self.loss_func(f2, torch.zeros_like(f2))
            loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()

            # 计算L2正则化项
            l2_reg = self.compute_l2_regularization()

            # 调整最终的总损失
            loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + l2_reg

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            l2_reg_meter.update(l2_reg.item())
            if (idx + 1) % 50 == 0:
                print(f"[epoch:{epoch} iter:{idx + 1}] data loss:{loss1_meter.avg:.6f}, "
                      f"PDE loss:{loss2_meter.avg:.6f}, physics loss:{loss3_meter.avg:.6f}, "
                      f"L2 loss:{l2_reg.item():.6f}")

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, l2_reg_meter.avg

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10

        # 创建一个列表来存储每个 epoch 的损失值
        epoch_losses = []

        for e in range(1, self.args.epochs + 1):
            early_stop += 1
            loss1, loss2, loss3, l2_reg = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()
            total_loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg
            epoch_losses.append(total_loss)  # 将总损失添加到列表中

            info = '[Train] epoch:{}, lr:{:.6f}, total loss:{:.6f}'.format(e, current_lr, total_loss)
            self.logger.info(info)

            # 每个epoch结束后记录注意力机制的变化
            epoch_attention_dir = os.path.join(self.attention_dir, f'epoch_{e}')
            self.solution_u.save_attention_data(epoch_attention_dir)
            self.dynamical_F.save_attention_data(os.path.join(epoch_attention_dir, 'dynamical_F'))

            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)

            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)

                    # 为最佳模型保存一份注意力权重
                    best_attention_dir = os.path.join(self.attention_dir, 'best_model')
                    self.solution_u.save_attention_data(best_attention_dir)
                    self.dynamical_F.save_attention_data(os.path.join(best_attention_dir, 'dynamical_F'))
                ##################################################################################

            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break

        # 保存所有epoch的损失
        if self.args.save_folder is not None:
            np.save(os.path.join(self.args.save_folder, 'epoch_losses.npy'), epoch_losses)

            # 在训练结束后绘制所有epoch的损失
            self._plot_epoch_losses(epoch_losses)

            # 还可以绘制所有epoch的注意力权重变化
            self._plot_attention_weights_evolution()

        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))

    def _plot_epoch_losses(self, epoch_losses):
        """绘制所有epoch的损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.title('Total Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.save_folder, 'epoch_losses.png'))
        plt.close()

    def _plot_attention_weights_evolution(self):
        """绘制所有epoch的注意力权重变化"""
        # 获取所有epoch目录
        epoch_dirs = sorted([d for d in os.listdir(self.attention_dir) if d.startswith('epoch_')],
                            key=lambda x: int(x.split('_')[1]))

        # 获取注意力层数量
        sample_dir = os.path.join(self.attention_dir, epoch_dirs[0])
        attention_files = [f for f in os.listdir(sample_dir) if f.endswith('_data.npy')]
        layer_ids = sorted(list(set([int(f.split('_')[2]) for f in attention_files])))

        # 为每个注意力层创建一个演化图
        for layer_id in layer_ids:
            weights_over_epochs = []
            for epoch_dir in epoch_dirs:
                file_path = os.path.join(self.attention_dir, epoch_dir, f'attention_layer_{layer_id}_data.npy')
                if os.path.exists(file_path):
                    data = np.load(file_path, allow_pickle=True).item()
                    weights_over_epochs.append(data['attention_weights'])

            if weights_over_epochs:
                # 创建热图
                plt.figure(figsize=(12, 8))
                weights_array = np.array(weights_over_epochs)
                sns.heatmap(weights_array, cmap='YlOrRd',
                            xticklabels=range(weights_array.shape[1]),
                            yticklabels=[d.split('_')[1] for d in epoch_dirs[:len(weights_over_epochs)]])
                plt.title(f'Attention Weights Evolution for Layer {layer_id}')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Epoch')
                plt.tight_layout()
                plt.savefig(os.path.join(self.attention_dir, f'layer_{layer_id}_weights_evolution.png'))
                plt.close()

                # 创建3D图
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                epochs = np.arange(len(weights_over_epochs))
                features = np.arange(weights_over_epochs[0].shape[0])
                X, Y = np.meshgrid(features, epochs)
                surf = ax.plot_surface(X, Y, weights_array, cmap='viridis', edgecolor='none', alpha=0.8)
                ax.set_title(f'Attention Weights Evolution for Layer {layer_id} (3D)')
                ax.set_xlabel('Feature Dimension')
                ax.set_ylabel('Epoch')
                ax.set_zlabel('Weight Value')
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                plt.tight_layout()
                plt.savefig(os.path.join(self.attention_dir, f'layer_{layer_id}_weights_evolution_3d.png'))
                plt.close()


if __name__ == "__main__":
    import argparse


    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=512, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
        parser.add_argument('--l2_lambda', type=float, default=1e-8, help='L2 regularization coefficient')
        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-3, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-3, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-2, help='learning rate of F')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

        parser.add_argument('--lambda1', type=float, default=1,
                            help='loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg')
        parser.add_argument('--lambda2', type=float, default=0.6,
                            help='loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg')
        parser.add_argument('--lambda3', type=float, default=3e-2,
                            help='loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg')
        parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()


    args = get_args()
    pinn = PINN(args)
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    print(pinn.dynamical_F)