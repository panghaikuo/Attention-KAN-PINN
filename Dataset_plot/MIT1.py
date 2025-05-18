import os
import pandas as pd
import numpy as np
import h5py
import pprint
import matplotlib.pyplot as plt
from scipy.stats import entropy, kurtosis, skew

class BatchBattery:
    def __init__(self,path):
        self.path = path
        self.f = h5py.File(path, 'r')
        self.batch = self.f['batch']
        self.date = self.f['batch_date'][:].tobytes()[::2].decode()
        self.num_cells = self.batch['summary'].shape[0]

        print('date: ',self.date)
        print('num_cells: ',self.num_cells)

    def get_one_battery(self,cell_num):
        '''
        读取一个电池的数据
        :param cell_num: 电池序号
        :return:
        '''
        i = cell_num
        f = self.f
        batch = self.batch
        cycles = f[batch['cycles'][i, 0]]
        cc_time, cv_time, cc_q, cv_q = [], [], [], []
        voltage_slope, voltage_entropy, voltage_kurtosis, voltage_skewness = [], [], [], []
        current_slope, current_entropy = [], []
        # 解析cycle的数据
        cycle_dict = {}
        for j in range(1, cycles['I'].shape[0]):
            I = np.hstack(self.f[cycles['I'][j, 0]][()])
            V = np.hstack(self.f[cycles['V'][j, 0]][()])
            Qc = np.hstack(self.f[cycles['Qc'][j, 0]][()])
            t = np.hstack(self.f[cycles['t'][j, 0]][()])

            cc_mask = (I >= 1)
            cc_time.append(np.sum(np.diff(t[cc_mask])))  # 转换为分钟
            cc_q.append(np.sum(Qc[cc_mask])/1000)

            # Calculate CV stage time and charge
            cv_mask = (V >= 3.6) & (I < 1)  # Define CV as voltage ≥ 3.6V and current < threshold for CC
            cv_time.append(np.sum(np.diff(t[cv_mask]) ))  # 转换为分钟
            cv_q.append(np.sum(Qc[cv_mask])/1000)

            # 计算特征值
            voltage_slope.append(np.mean(np.gradient(V)))
            voltage_entropy.append(entropy(np.histogram(V, bins=10)[0] + 1))
            voltage_kurtosis.append(kurtosis(V))
            voltage_skewness.append(skew(V))

            current_slope.append(np.mean(np.gradient(I)))
            current_entropy.append(entropy(np.histogram(I, bins=10)[0] + 1))
            summary={

                'CC_time': cc_time,              # 电池的恒流充电时间
                'CV_time': cv_time,              # 电池的恒压充电时间
                'CC_Q': cc_q,                    # 电池的恒流充电电量
                'CV_Q': cv_q,                    # 电池的恒压充电电量
                'voltage_slope': voltage_slope,        # 电池的电压斜率
                'voltage_entropy': voltage_entropy,    # 电池的电压熵
                'voltage_kurtosis': voltage_kurtosis,     # 电池的电压峰度
                'voltage_skewness': voltage_skewness,      # 电池的电压偏度
                'current_slope': current_slope,          # 电池的电流斜率
                'current_entropy': current_entropy        # 电池的电流熵

            }
        barcode_dataset = self.f[batch['barcode'][i, 0]]
        barcode = barcode_dataset[()].tobytes()[::2].decode() \
            if isinstance(barcode_dataset[()], np.ndarray) \
            else \
            barcode_dataset[()].decode()
        try:
            channel_ID = self.f[batch['channel_id'][i, 0]][()].tobytes().decode('utf-8')
        except UnicodeDecodeError:
            # 尝试使用'utf-16-le'或'utf-16-be'进行解码，根据数据实际情况选择
            try:
                channel_ID = self.f[batch['channel_id'][i, 0]][()].tobytes().decode('utf-16-le')[::2]
            except UnicodeDecodeError:
                channel_ID = self.f[batch['channel_id'][i, 0]][()].tobytes().decode('utf-16-be')[::2]
        summary_IR = np.hstack(self.f[batch['summary'][i, 0]]['IR'][0, :].tolist())     #电池内阻的汇总数据，通常是电池在不同循环或状态下的内阻测量值。
        summary_QC = np.hstack(self.f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())   #电池充电过程中传输的电量（充电量），以安时（Ah）为单位。这显示了电池在充电时所能存储的电量。
        summary_TA = np.hstack(self.f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())     #电池的平均温度数据，表示在充电或放电过程中电池的温度变化情况。
        summary_TM = np.hstack(self.f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())     #电池的最低温度，表示在充电或放电过程中记录到的最低温度。
        summary_TX = np.hstack(self.f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())     #电池的最高温度，表示在充电或放电过程中记录到的最高温度。
        summary_CT = np.hstack(self.f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())   #电池的充电时间，表示完成一次充电所需的时间，通常以分钟为单位。
        summary_QD = np.hstack(self.f[batch['summary'][i, 0]]['QDischarge'][0, 1:].tolist())    #电池放电过程中传输的电量（放电量），以安时（Ah）为单位。这反映了电池在放电时所能释放的电量。

        summary.update({'IR':summary_IR,'QC':summary_QC,'TA':summary_TA,'TM':summary_TM,'TX':summary_TX, 'CT':summary_CT,'capacity':summary_QD})


        return summary,cycle_dict

    def get_one_battery_one_cycle(self,cell_num,cycle_num):
        '''
        读取一个电池的某个cycle的数据
        :param cell_num: 电池序号
        :param cycle_num: cycle序号
        :return: DataFrame
        '''
        i = cell_num
        f = self.f
        batch = self.batch

        cycles = f[batch['cycles'][i, 0]]
        # 检查cycle_num是否合法
        if cycle_num >= cycles['I'].shape[0] or cycle_num < 1:
            raise ValueError('cycle_num must be in [1,{}]'.format(cycles['I'].shape[0]))

        # 解析cycle的数据
        j = cycle_num
        I = np.hstack((f[cycles['I'][j, 0]][:]))  # 修改访问方式
        V = np.hstack((f[cycles['V'][j, 0]][:]))  # 修改访问方式
        Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))  # 修改访问方式
        Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))  # 修改访问方式
        t = np.hstack((f[cycles['t'][j, 0]][:]))  # 修改访问方式
        # internal_resistance = V / np.where(I == 0, np.nan, I)  # 计算内阻, 'internal_resistance (Ω)': internal_resistance
        one_cycle = {'current (A)': I, 'voltage (V)': V, 'charge Q (Ah)': Qc,
                     'discharge Q (Ah)': Qd, 'time (min)': t}
        cycle_df = pd.DataFrame(one_cycle)
        return cycle_df



    def save_battery_summary_to_csv(self, cell_num, filename):
        directory = os.path.dirname(filename)

        # 如果目录不存在，则创建该目录
        if not os.path.exists(directory):
            os.makedirs(directory)
        summary, _ = self.get_one_battery(cell_num)
        # 确保summary字典中的所有值都是可迭代的（例如列表或元组）
        # 这里假设我们需要将整数转换为包含单个元素的列表
        summary = {k: [v] if isinstance(v, int) else v for k, v in summary.items()}
        # 将summary字典转换为DataFrame
        summary_df = pd.DataFrame.from_dict(summary, orient='index').T
        summary_df.to_csv(filename, index=False)
        print(f'Saved battery summary to {filename}')






if __name__ == '__main__':
    path = '../2017-06-30_batchdata_updated_struct_errorcorrect.mat'
    battery = int(input("请输入电池序号: "))  # 用户输入电池序号
    filename = f'../1/2017-06-30_battery-{battery}.csv'
    bb = BatchBattery(path)
    bb.save_battery_summary_to_csv(battery, filename)  # 保存电池特征到CSV



