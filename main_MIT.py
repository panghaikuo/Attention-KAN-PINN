
from dataloader.dataloader import MyMITdata
from Model.Model_MLPtoKANs import PINN
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for MIT dataset')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-6, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')


    # model related
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--lambda1', type=float, default=1,
                        help='loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg')
    parser.add_argument('--lambda2', type=float, default=0.6,
                        help='loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg')
    parser.add_argument('--lambda3', type=float, default=1e-2,
                        help='loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='MIT results', help='save folder')

    args = parser.parse_args()

    return args

def load_MyMIT_data(args):
    root = 'Dataset_plot/My data/MIT data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))
    data = MyMITdata(root=root,args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader
def main():
    args = get_args()

    # 创建基础结果目录
    base_folder = 'MIT_sensitivity_analysis1'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # 单参数敏感性分析 - lambda1
    analyze_lambda1(args, base_folder)

    # 单参数敏感性分析 - lambda2
    analyze_lambda2(args, base_folder)

    # 单参数敏感性分析 - lambda3
    analyze_lambda3(args, base_folder)

    # 单参数敏感性分析 - l2_lambda
    analyze_l2_lambda(args, base_folder)

    # 组合敏感性分析 - lambda1 和 lambda2
    analyze_lambda1_lambda2(args, base_folder)

    # 组合敏感性分析 - lambda2 和 lambda3
    analyze_lambda2_lambda3(args, base_folder)


def analyze_lambda1(args, base_folder):
    """分析lambda1的敏感性"""
    folder = os.path.join(base_folder, 'lambda1_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 测试一系列lambda1值
    lambda1_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for value in lambda1_values:
        # 设置实验子目录
        exp_folder = os.path.join(folder, f'lambda1_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        # 修改参数
        setattr(args, 'lambda1', value)
        setattr(args, 'save_folder', exp_folder)

        # 加载数据并训练模型
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

        # 保存此实验的参数设置
        save_experiment_config(args, exp_folder)


def analyze_lambda2(args, base_folder):
    """分析lambda2的敏感性"""
    folder = os.path.join(base_folder, 'lambda2_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 测试一系列lambda2值
    lambda2_values = [0.1, 0.3, 0.6, 1.0, 2.0]

    for value in lambda2_values:
        # 设置实验子目录
        exp_folder = os.path.join(folder, f'lambda2_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        # 修改参数
        setattr(args, 'lambda2', value)
        setattr(args, 'save_folder', exp_folder)

        # 加载数据并训练模型
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

        # 保存此实验的参数设置
        save_experiment_config(args, exp_folder)


def analyze_lambda3(args, base_folder):
    """分析lambda3的敏感性"""
    folder = os.path.join(base_folder, 'lambda3_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 测试一系列lambda3值 - 使用科学计数法表示小数
    lambda3_values = [1e-3, 5e-3, 1e-2, 3e-2, 5e-2]

    for value in lambda3_values:
        # 设置实验子目录
        exp_folder = os.path.join(folder, f'lambda3_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        # 修改参数
        setattr(args, 'lambda3', value)
        setattr(args, 'save_folder', exp_folder)

        # 加载数据并训练模型
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

        # 保存此实验的参数设置
        save_experiment_config(args, exp_folder)


def analyze_l2_lambda(args, base_folder):
    """分析l2_lambda的敏感性"""
    folder = os.path.join(base_folder, 'l2_lambda_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 测试一系列l2_lambda值
    l2_lambda_values = [1e-6, 1e-5, 1e-4, 1e-3]

    for value in l2_lambda_values:
        # 设置实验子目录
        exp_folder = os.path.join(folder, f'l2_lambda_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        # 修改参数
        setattr(args, 'l2_lambda', value)
        setattr(args, 'save_folder', exp_folder)

        # 加载数据并训练模型
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

        # 保存此实验的参数设置
        save_experiment_config(args, exp_folder)


def analyze_lambda1_lambda2(args, base_folder):
    """分析lambda1和lambda2组合的敏感性"""
    folder = os.path.join(base_folder, 'lambda1_lambda2_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 测试lambda1和lambda2的组合
    lambda1_values = [0.5, 1.0, 2.0]
    lambda2_values = [0.3, 0.6, 1.0]

    for l1 in lambda1_values:
        for l2 in lambda2_values:
            # 设置实验子目录
            exp_folder = os.path.join(folder, f'lambda1_{l1}_lambda2_{l2}')
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)

            # 修改参数
            setattr(args, 'lambda1', l1)
            setattr(args, 'lambda2', l2)
            setattr(args, 'save_folder', exp_folder)

            # 加载数据并训练模型
            dataloader = load_MyMIT_data(args)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

            # 保存此实验的参数设置
            save_experiment_config(args, exp_folder)


def analyze_lambda2_lambda3(args, base_folder):
    """分析lambda2和lambda3组合的敏感性"""
    folder = os.path.join(base_folder, 'lambda2_lambda3_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 测试lambda2和lambda3的组合
    lambda2_values = [0.3, 0.6, 1.0]
    lambda3_values = [1e-2, 3e-2, 5e-2]

    for l2 in lambda2_values:
        for l3 in lambda3_values:
            # 设置实验子目录
            exp_folder = os.path.join(folder, f'lambda2_{l2}_lambda3_{l3}')
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)

            # 修改参数
            setattr(args, 'lambda2', l2)
            setattr(args, 'lambda3', l3)
            setattr(args, 'save_folder', exp_folder)

            # 加载数据并训练模型
            dataloader = load_MyMIT_data(args)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

            # 保存此实验的参数设置
            save_experiment_config(args, exp_folder)


def save_experiment_config(args, folder):
    """保存实验配置参数到文件"""
    config_path = os.path.join(folder, 'config.txt')
    with open(config_path, 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f"{k}: {v}\n")

if __name__ == '__main__':
    #pass
    main()
