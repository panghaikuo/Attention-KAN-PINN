import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于张量操作和深度学习
from torch.utils.data import TensorDataset  # 导入TensorDataset，用于处理张量数据
from torch.utils.data import DataLoader  # 导入DataLoader，用于批处理数据
import os  # 导入os库，用于文件和目录操作
import random  # 导入random库，用于生成随机数
from sklearn.model_selection import train_test_split  # 导入train_test_split，用于数据分割
from utils.util import write_to_txt  # 导入自定义工具函数，用于写入文本文件


class DF():  # 定义DF类
    def __init__(self, args):  # 构造函数，接收参数
        self.normalization = True  # 设置归一化标志为True
        self.normalization_method = args.normalization_method  # 获取归一化方法（min-max或z-score）
        self.args = args  # 保存其他参数

    def _3_sigma(self, Ser1):  # 定义计算三倍标准差的函数
        '''
        计算超出三倍标准差的值的索引
        :param Ser1: 输入序列（如DataFrame的一列）
        :return: 超出范围的索引
        '''
        # 根据均值和标准差计算异常值条件
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]  # 获取满足条件的索引
        return index  # 返回异常值索引

    def delete_3_sigma(self, df):  # 定义删除异常值的函数
        '''
        根据三倍标准差规则删除DataFrame中的异常值
        :param df: 输入DataFrame
        :return: 删除异常值后的DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)  # 将无穷大替换为NaN
        df = df.dropna()  # 删除含有NaN的行
        df = df.reset_index(drop=True)  # 重置DataFrame的索引
        out_index = []  # 初始化用于存储异常值索引的列表
        for col in df.columns:  # 遍历DataFrame的每一列
            index = self._3_sigma(df[col])  # 获取该列的异常值索引
            out_index.extend(index)  # 将异常值索引添加到列表中
        out_index = list(set(out_index))  # 去重
        df = df.drop(out_index, axis=0)  # 删除异常值所在的行
        df = df.reset_index(drop=True)  # 再次重置索引
        return df  # 返回处理后的DataFrame

    def read_one_csv(self, file_name, nominal_capacity=None):  # 定义读取CSV文件的函数
        '''
        读取一个CSV文件并返回清洗后的DataFrame
        :param file_name: 文件名（字符串）
        :param nominal_capacity: 名义容量（可选）
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)  # 读取CSV文件
        df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0]))  # 插入循环索引列

        df = self.delete_3_sigma(df)  # 删除异常值

        if nominal_capacity is not None:  # 如果提供了名义容量
            # print(f'nominal_capacity:{nominal_capacity}, capacity max:{df["capacity"].max()}', end=',')
            df['capacity'] = df['capacity'] / nominal_capacity  # 计算相对容量
            # print(f'SOH max:{df["capacity"].max()}')
            f_df = df.iloc[:, :-1]  # 获取去掉最后一列的数据
            if self.normalization_method == 'min-max':  # 根据选择的归一化方法进行归一化
                f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1  # min-max归一化
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean()) / f_df.std()  # z-score归一化

            df.iloc[:, :-1] = f_df  # 将归一化后的数据更新回DataFrame

        return df  # 返回处理后的DataFrame

    def load_one_battery(self, path, nominal_capacity=None):  # 定义加载一个电池数据的函数
        '''
        读取一个CSV文件并将数据分为x和y
        :param path: 文件路径
        :param nominal_capacity: 名义容量（可选）
        :return: x和y的元组
        '''
        df = self.read_one_csv(path, nominal_capacity)  # 读取CSV文件
        x = df.iloc[:, :-1].values  # 获取特征数据
        y = df.iloc[:, -1].values  # 获取标签数据
        x1 = x[:-1]  # 获取前n-1个特征
        x2 = x[1:]  # 获取后n-1个特征
        y1 = y[:-1]  # 获取前n-1个标签
        y2 = y[1:]  # 获取后n-1个标签
        return (x1, y1), (x2, y2)  # 返回特征和标签的元组

    def load_all_battery(self, path_list, nominal_capacity):  # 定义加载多个电池数据的函数
        '''
        读取多个CSV文件，将数据分为X和Y，并将其打包到dataloader中
        :param path_list: 文件路径列表
        :param nominal_capacity: 名义容量，用于计算SOH
        :return: Dataloader
        '''
        X1, X2, Y1, Y2 = [], [], [], []  # 初始化特征和标签列表
        if hasattr(self.args, 'log_dir') and hasattr(self.args, 'save_folder'):  # 检查参数中是否有日志目录和保存目录
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)  # 拼接保存路径
            write_to_txt(save_name, 'data path:')  # 写入数据路径到文本文件
            write_to_txt(save_name, str(path_list))  # 写入路径列表到文本文件
        for path in path_list:  # 遍历路径列表
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)  # 读取每个电池数据
            # print(path)
            # print(x1.shape, x2.shape, y1.shape, y2.shape)
            X1.append(x1)  # 添加到特征列表X1
            X2.append(x2)  # 添加到特征列表X2
            Y1.append(y1)  # 添加到标签列表Y1
            Y2.append(y2)  # 添加到标签列表Y2

        # 将所有特征和标签连接为一个数组
        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        # 转换为PyTorch张量
        tensor_X1 = torch.from_numpy(X1).float()  # 将特征转换为浮点型张量
        tensor_X2 = torch.from_numpy(X2).float()  # 将特征转换为浮点型张量
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)  # 将标签形状调整为(n, 1)并转换为浮点型张量
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1, 1)  # 将标签形状调整为(n, 1)并转换为浮点型张量
        # print('X shape:', tensor_X1.shape)
        # print('Y shape:', tensor_Y1.shape)

        # Condition 1
        # 1.1 划分训练集和测试集
        split = int(tensor_X1.shape[0] * 0.8)  # 计算分割点
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]  # 划分训练集和测试集
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
        # 1.2 划分训练集和验证集
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)  # 划分训练集和验证集

        # 创建数据加载器
        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,  # 设置批大小
                                  shuffle=True)  # 随机打乱数据
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,  # 设置批大小
                                  shuffle=True)  # 随机打乱数据
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size,  # 设置批大小
                                 shuffle=False)  # 不打乱数据

        # Condition 2
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)  # 再次划分训练集和验证集
        train_loader_2 = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,  # 设置批大小
                                  shuffle=True)  # 随机打乱数据
        valid_loader_2 = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,  # 设置批大小
                                  shuffle=True)  # 随机打乱数据

        # Condition 3
        test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                 batch_size=self.args.batch_size,  # 设置批大小
                                 shuffle=False)  # 不打乱数据

        # 返回加载器字典
        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2, 'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader  # 返回数据加载器


class MyMITdata(DF):  # 定义MITdata类，继承自DF类
    def __init__(self, root='../Dataset_plot/My data/MIT data', args=None):  # 构造函数
        super(MyMITdata, self).__init__(args)  # 调用父类构造函数
        self.root = root  # 保存数据根目录
        self.batchs = ['2017-05-12', '2017-06-30', '2018-04-12']  # 批次名称
        if self.normalization:  # 如果进行归一化
            self.nominal_capacity = 1.1  # 设置名义容量
        else:
            self.nominal_capacity = None  # 不使用名义容量
        # print('-' * 20, 'MIT data', '-' * 20)

    def read_one_batch(self, batch):  # 定义读取一个批次数据的函数
        '''
        读取一个批次的CSV文件
        :param batch: int, 可选[1,2,3]
        :return: dict
        '''
        assert batch in [1, 2, 3], 'batch must be in {}'.format([1, 2, 3])  # 检查批次有效性
        root = os.path.join(self.root, self.batchs[batch - 1])  # 获取批次目录路径
        file_list = os.listdir(root)  # 获取该批次的所有文件
        path_list = []  # 初始化路径列表
        for file in file_list:  # 遍历文件
            file_name = os.path.join(root, file)  # 拼接文件路径
            path_list.append(file_name)  # 添加到路径列表
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)  # 加载电池数据

    def read_all(self, specific_path_list=None):  # 定义读取所有数据的函数
        '''
        读取所有CSV文件。如果指定了specific_path_list，则读取指定的文件；否则读取所有文件；
        :param specific_path_list: 可选的特定文件路径列表
        :return: dict
        '''
        if specific_path_list is None:  # 如果没有指定特定路径
            file_list = []  # 初始化文件列表
            for batch in self.batchs:  # 遍历所有批次
                root = os.path.join(self.root, batch)  # 获取批次目录路径
                files = os.listdir(root)  # 获取该批次的所有文件
                for file in files:  # 遍历文件
                    path = os.path.join(root, file)  # 拼接文件路径
                    file_list.append(path)  # 添加到文件列表
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)  # 加载电池数据
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)  # 加载特定路径的电池数据


class MyTJUdata(DF):
    def __init__(self,root='Dataset_plot/My data/TJU data',args=None):
        super(MyTJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]
        #print('-' * 20, 'TJU data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]; optional[1,2,3]
        :return: DataFrame
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        df = pd.DataFrame()
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch])

    def read_all(self,specific_path_list):
        '''
        读取所有csv文件,封装成dataloader
        English version: Read all csv files and encapsulate them into a dataloader
        :param self:
        :return: dict
        '''
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)


if __name__ == '__main__':  # 如果是主程序
    import argparse  # 导入argparse库用于解析命令行参数
    def get_args():  # 定义获取命令行参数的函数
        parser = argparse.ArgumentParser()  # 创建参数解析器
        parser.add_argument('--data', type=str, default='MyMIT', help='数据集名称 TJU, MyMIT')  # 添加数据集参数
        parser.add_argument('--batch', type=int, default=1, help='批次编号：1,2,3')  # 添加批次参数
        parser.add_argument('--batch_size', type=int, default=256, help='批大小')  # 添加批大小参数
        parser.add_argument('--normalization_method', type=str, default='min-max', help='归一化方法')  # 添加归一化方法参数
        parser.add_argument('--log_dir', type=str, default='test.txt', help='日志文件目录')  # 添加日志目录参数
        return parser.parse_args()  # 返回解析后的参数

    args = get_args()  # 获取命令行参数


    mit = MyMITdata(args=args)  # 创建MIT数据类实例
    mit.read_one_batch(batch=1)  # 读取指定批次
    loader = mit.read_all()  # 读取所有数据

    train_loader = loader['train']  # 获取训练加载器
    test_loader = loader['test']  # 获取测试加载器
    valid_loader = loader['valid']  # 获取验证加载器
    all_loader = loader['test_3']  # 获取所有加载器
    print('train_loader:', len(train_loader), 'test_loader:', len(test_loader), 'valid_loader:', len(valid_loader), 'all_loader:', len(all_loader))  # 输出加载器的长度

    for iter, (x1, x2, y1, y2) in enumerate(train_loader):  # 遍历训练加载器
        print('x1 shape:', x1.shape)  # 输出x1的形状
        print('x2 shape:', x2.shape)  # 输出x2的形状
        print('y1 shape:', y1.shape)  # 输出y1的形状
        print('y2 shape:', y2.shape)  # 输出y2的形状
        print('y1 max:', y1.max())  # 输出y1的最大值
        break  # 只输出一次，退出循环
