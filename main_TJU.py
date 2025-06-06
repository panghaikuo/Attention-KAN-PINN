
from dataloader.dataloader import MyTJUdata
from Model.Model_MLPtoKANs import PINN
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for TJU dataset')
    parser.add_argument('--data', type=str, default='MyTJU', help='XJTU, HUST, MIT, MyTJU')
    parser.add_argument('--in_same_batch', type=bool, default=True, help='训练集和测试集是否在同一个batch中(whether train and test sets are in the same batch)')
    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=0, choices=[-1,0,1,2],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

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
    parser.add_argument('--save_folder', type=str, default='MyTJU results', help='save folder')

    args = parser.parse_args()

    return args


def load_TJU_data(args):
    root = 'data/TJU data'
    data = MyTJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # 序号的个位数字是5或者9的是测试集，其他的是训练集
    # The numbers whose units digit is 5 or 9 are test set, and the others are training set
    mod = [(9,19),(5,19),(5,9)]
    if args.in_same_batch:
        batchs = os.listdir(root)
        batch = batchs[args.batch]
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for i,f in enumerate(files):
            # 判断i的个位数字是否为5或者9 (judge whether the units digit of i is 5 or 9)
            id = i + 1
            if id % 10 == mod[args.batch][0] or id % 10 == mod[args.batch][1]:
                test_list.append(os.path.join(batch_root,f))
                print(f)
            else:
                train_list.append(os.path.join(batch_root,f))
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    else: # 如果训练集和测试集不在同一个batch中，则一个batch用来训练，另一个batch用来测试
        # (If the training set and test set are not in the same batch,
        # one batch is used for training and the other batch is used for testing)
        batchs = os.listdir(root)
        train_loader = data.read_one_batch(args.train_batch)
        test_loader = data.read_one_batch(args.test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader



def main():
    args = get_args()
    batchs = [0]

    for batch in batchs:
        setattr(args, 'in_same_batch', True)
        setattr(args, 'batch', batch)
        for e in range(10):
            if args.in_same_batch:
                save_folder = 'MyTJU results/TJU-Attention-KAN-PINN0-0_7 results/' + str(batch) + '-' + str(batch) + '/Experiment' + str(e + 1)
            else:
                save_folder = 'MyTJU results/TJU-Attention-KAN-PINN0-0_7 results/' + str(args.train_batch) + '-' + str(
                    args.test_batch) + '/Experiment' + str(e + 1)
            setattr(args, "save_folder", save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            dataloader = load_TJU_data(args)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

if __name__ == '__main__':
    main()


