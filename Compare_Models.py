import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from Model.Model_MLPtoKANs import KAN as Encoder  # 从Model.Model导入MLP作为Encoder
from Model.Model_MLPtoKANs import Predictor  # 从Model.Model导入预测器

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断是否使用GPU，如果可用则使用CUDA，否则使用CPU


class ResBlock(nn.Module):  # 定义残差块类
    def __init__(self, input_channel, output_channel, stride):  # 初始化方法，接收输入通道、输出通道和步幅
        super(ResBlock, self).__init__()  # 调用父类的初始化方法
        self.conv = nn.Sequential(  # 定义卷积层序列
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),  # 卷积层
            nn.BatchNorm1d(output_channel),  # 批归一化层
            nn.ReLU(),  # 激活函数ReLU

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),  # 第二个卷积层
            nn.BatchNorm1d(output_channel)  # 第二个批归一化层
        )

        self.skip_connection = nn.Sequential()  # 初始化跳跃连接
        if output_channel != input_channel:  # 如果输入和输出通道不同
            self.skip_connection = nn.Sequential(  # 定义跳跃连接的卷积层
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)  # 批归一化层
            )

        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, x):  # 前向传播方法
        out = self.conv(x)  # 通过卷积层计算输出
        out = self.skip_connection(x) + out  # 加上跳跃连接的输出
        out = self.relu(out)  # 激活
        return out  # 返回输出


class MLP(nn.Module):  # 定义多层感知机类
    def __init__(self):  # 初始化方法
        super(MLP, self).__init__()  # 调用父类的初始化方法
        self.encoder = Encoder(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, dropout=0.2)  # 初始化编码器
        self.predictor = Predictor(input_dim=32)  # 初始化预测器

    def forward(self, x):  # 前向传播方法
        x = self.encoder(x)  # 通过编码器处理输入
        x = self.predictor(x)  # 通过预测器处理输出
        return x  # 返回输出


class LSTM(nn.Module):  # 定义LSTM类
    def __init__(self, input_dim=17, hidden_dim=60, output_dim=1, num_layers=2, dropout=0.2):  # 初始化方法
        super(LSTM, self).__init__()  # 调用父类的初始化方法
        self.input_dim = input_dim  # 输入维度
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层数
        self.output_dim = output_dim  # 输出维度
        self.device = device  # 保存设备信息

        # 定义LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)  # LSTM层

        # 定义用于预测的全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层
        for layer in self.modules():  # 遍历模型中的所有层
            if isinstance(layer, nn.Linear):  # 如果是线性层
                nn.init.xavier_normal_(layer.weight)  # 使用Xavier初始化权重
                nn.init.constant_(layer.bias, 0)  # 将偏置初始化为0

    def forward(self, x):  # 前向传播方法
        x = x.to(self.device)  # 确保输入张量在正确的设备上

        # 初始化隐藏状态和细胞状态，确保它们在与输入相同的设备上
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 细胞状态

        if x.dim() == 2:  # 如果输入是二维的
            x = x.unsqueeze(1)  # 添加时间维度

        # 前向传播通过LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: 形状为(batch_size, seq_length, hidden_size)的张量

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])  # 通过全连接层预测
        return out  # 返回输出


class CNN(nn.Module):  # 定义卷积神经网络类
    def __init__(self):  # 初始化方法
        super(CNN, self).__init__()  # 调用父类的初始化方法
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1)  # 第一层残差块
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)  # 第二层残差块
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2)  # 第三层残差块
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)  # 第四层残差块
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1)  # 第五层残差块
        self.layer6 = nn.Linear(8 * 5, 1)  # 最后一层全连接层

    def forward(self, x):  # 前向传播方法
        N, L = x.shape[0], x.shape[1]  # 获取批次大小和序列长度
        x = x.view(N, 1, L)  # 调整输入形状
        out = self.layer1(x)  # 通过第一层
        out = self.layer2(out)  # 通过第二层
        out = self.layer3(out)  # 通过第三层
        out = self.layer4(out)  # 通过第四层
        out = self.layer5(out)  # 通过第五层
        out = self.layer6(out.view(N, -1))  # 通过全连接层
        return out.view(N, 1)  # 返回形状为(N, 1)的输出


def count_parameters(model):  # 计算模型参数数量的函数
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计可训练参数的数量
    print('The model has {} trainable parameters'.format(count))  # 打印参数数量


if __name__ == '__main__':  # 主程序入口
    x = torch.randn(10, 17)  # 随机生成输入

    # 为所有模型指定相同的设备
    mlp_model = MLP().to(device)
    cnn_model = CNN().to(device)
    lstm_model = LSTM().to(device)

    # 确保输入数据在正确的设备上
    x = x.to(device)

    y1 = mlp_model(x)  # 使用MLP模型
    y2 = cnn_model(x)  # 使用CNN模型
    y3 = lstm_model(x)  # 使用LSTM模型

    count_parameters(mlp_model)  # 计算MLP模型参数数量
    count_parameters(cnn_model)  # 计算CNN模型参数数量
    count_parameters(lstm_model)  # 计算LSTM模型参数数量