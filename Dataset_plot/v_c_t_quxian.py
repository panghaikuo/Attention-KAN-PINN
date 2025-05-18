import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib.ticker import MaxNLocator


class BatchBattery:
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, 'r')
        self.batch = self.f['batch']
        self.date = self.f['batch_date'][:].tobytes()[::2].decode()
        self.num_cells = self.batch['summary'].shape[0]

        print('date: ', self.date)
        print('num_cells: ', self.num_cells)

    def get_one_battery(self, cell_num):
        '''
        读取一个电池的数据
        :param cell_num: 电池序号
        :return:
        '''
        i = cell_num
        f = self.f
        batch = self.batch
        summary_TM = np.hstack(self.f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())  # 电池的最低温度
        summary_TX = np.hstack(self.f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())  # 电池的最高温度
        summary = {'TM': summary_TM, 'TX': summary_TX}
        cycles = f[batch['cycles'][i, 0]]

        cycle_dict = {}
        for j in range(1, cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j, 0]][()]))  # 电流
            V = np.hstack((f[cycles['V'][j, 0]][()]))  # 电压
            Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))  # 充电电量
            Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))  # 放电电量
            t = np.hstack((f[cycles['t'][j, 0]][()]))  # 时间
            one_cycle = {'current (A)': I, 'voltage (V)': V, 'charge Q (Ah)': Qc,
                         'discharge Q (Ah)': Qd, 'time (min)': t}
            cycle_dict[j] = one_cycle

        return summary, cycle_dict

    def get_one_battery_one_cycle(self, cell_num, cycle_num):
        '''
        读取某个电池某个循环的数据
        :param cell_num: 电池序号
        :param cycle_num: 循环序号
        :return: DataFrame
        '''
        i = cell_num
        f = self.f
        batch = self.batch
        cycles = f[batch['cycles'][i, 0]]

        # 检查循环是否合法
        if cycle_num >= cycles['I'].shape[0] or cycle_num < 1:
            raise ValueError('cycle_num must be in [1,{}]'.format(cycles['I'].shape[0]))

        j = cycle_num
        I = np.hstack((f[cycles['I'][j, 0]][()]))
        V = np.hstack((f[cycles['V'][j, 0]][()]))
        Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
        Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
        t = np.hstack((f[cycles['t'][j, 0]][()]))

        one_cycle = {'current (A)': I, 'voltage (V)': V, 'charge Q (Ah)': Qc,
                     'discharge Q (Ah)': Qd, 'time (min)': t}
        cycle_df = pd.DataFrame(one_cycle)
        return cycle_df

    def plot_one_battery_one_cycle(self, cell_num, cycle_num):
        '''
        画出某个电池的一个循环的所有曲线，并加粗曲线和轴标签
        :param cell_num: 电池序号
        :param cycle_num: 循环序号
        :return: None
        '''
        cycle_df = self.get_one_battery_one_cycle(cell_num, cycle_num)
        summary, _ = self.get_one_battery(cell_num)
        summary_TM = summary['TM']  # 电池的最低温度
        summary_TX = summary['TX']  # 电池的最高温度

        # 确保温度数据与时间数据长度一致，使用插值方法
        time_length = len(cycle_df['time (min)'])
        time_points = cycle_df['time (min)']

        # 插值最低和最高温度数据到相同长度
        interp_func_TM = interp1d(np.linspace(0, 1, len(summary_TM)), summary_TM, kind='linear')
        interp_func_TX = interp1d(np.linspace(0, 1, len(summary_TX)), summary_TX, kind='linear')
        summary_TM_interp = interp_func_TM(np.linspace(0, 1, time_length))
        summary_TX_interp = interp_func_TX(np.linspace(0, 1, time_length))

        # 使用 Savitzky-Golay 滤波器平滑温度曲线
        smoothed_TM = savgol_filter(summary_TM_interp, window_length=51, polyorder=3)
        smoothed_TX = savgol_filter(summary_TX_interp, window_length=51, polyorder=3)

        # 绘制最低温度曲线
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(time_points, smoothed_TM, c='orange', linewidth=4)
        plt.ylabel('Min Temperature (°C)', fontsize=12, fontweight='bold')
        plt.ylim([27, 29])  # 设置最低温度范围
        plt.xlabel('Time (min)', fontsize=12, fontweight='bold')
        plt.tick_params(axis='both', labelsize=12, width=2)  # 加粗刻度线和标签
        plt.gca().spines['top'].set_linewidth(2)  # 加粗顶部框
        plt.gca().spines['right'].set_linewidth(2)  # 加粗右侧框
        plt.gca().spines['bottom'].set_linewidth(2)  # 加粗底部框
        plt.gca().spines['left'].set_linewidth(2)  # 加粗左侧框
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # 减少y轴刻度数量
        # 删除 "F12" 标注
        plt.tight_layout()
        plt.savefig('min_temperature_plot.png', dpi=300)  # 保存第一幅图
        plt.show()

        # 绘制最高温度曲线
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(time_points, smoothed_TX, c='blue', linewidth=4)
        plt.ylabel('Max Temperature (°C)', fontsize=12, fontweight='bold')
        plt.ylim([31, 34])  # 设置最高温度范围
        plt.xlabel('Time (min)', fontsize=12, fontweight='bold')
        plt.tick_params(axis='both', labelsize=12, width=2)  # 加粗刻度线和标签
        plt.gca().spines['top'].set_linewidth(2)  # 加粗顶部框
        plt.gca().spines['right'].set_linewidth(2)  # 加粗右侧框
        plt.gca().spines['bottom'].set_linewidth(2)  # 加粗底部框
        plt.gca().spines['left'].set_linewidth(2)  # 加粗左侧框
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # 减少y轴刻度数量
        # 删除 "F13" 标注
        plt.tight_layout()
        plt.savefig('max_temperature_plot.png', dpi=300)  # 保存第二幅图
        plt.show()

        # 绘制电流变化曲线
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(cycle_df['time (min)'], cycle_df['current (A)'], c='r', linewidth=4)
        plt.ylabel('Current (A)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (min)', fontsize=12, fontweight='bold')

        # 删除虚线
        # plt.axvline(15, color='black', linestyle='--', linewidth=2.5)  # 起始虚线
        # plt.axvline(23.8, color='black', linestyle='--', linewidth=2.5)  # 结束虚线
        # 删除 "F2" 标注
        plt.tick_params(axis='both', labelsize=12, width=2)  # 加粗刻度线和标签
        plt.gca().spines['top'].set_linewidth(2)  # 加粗顶部框
        plt.gca().spines['right'].set_linewidth(2)  # 加粗右侧框
        plt.gca().spines['bottom'].set_linewidth(2)  # 加粗底部框
        plt.gca().spines['left'].set_linewidth(2)  # 加粗左侧框
        plt.tight_layout()
        plt.savefig('C.png', dpi=300)
        plt.show()

        # 绘制电压变化曲线
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(cycle_df['time (min)'], cycle_df['voltage (V)'], c='g', linewidth=4)
        plt.ylabel('Voltage (V)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (min)', fontsize=12, fontweight='bold')

        # 删除虚线
        # plt.axvline(23.5, color='black', linestyle='--', linewidth=2.5)  # 起始虚线
        # plt.axvline(35.3, color='black', linestyle='--', linewidth=2.5)  # 结束虚线
        # 删除 "F3" 标注
        plt.tick_params(axis='both', labelsize=12, width=2)  # 加粗刻度线和标签
        plt.gca().spines['top'].set_linewidth(2)  # 加粗顶部框
        plt.gca().spines['right'].set_linewidth(2)  # 加粗右侧框
        plt.gca().spines['bottom'].set_linewidth(2)  # 加粗底部框
        plt.gca().spines['left'].set_linewidth(2)  # 加粗左侧框
        plt.tight_layout()
        plt.savefig('V.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    path = '../2017-06-30_batchdata_updated_struct_errorcorrect.mat'
    battery = 0
    cycle_num = 1

    bb = BatchBattery(path)
    summary, cycle = bb.get_one_battery(battery)

    bb.plot_one_battery_one_cycle(battery, cycle_num)
