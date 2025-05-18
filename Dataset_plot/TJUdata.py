import numpy as np
import pandas as pd
import os
from scipy.stats import entropy, kurtosis, skew
from sklearn.linear_model import LinearRegression
import glob

class Battery:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)

        # More robust filename parsing
        file_name = os.path.basename(path)
        self.original_file_name = file_name

        # Set defaults in case parsing fails
        self.temperature = 25  # Default temperature
        self.charge_c_rate = "1"  # Default charge rate
        self.discharge_c_rate = "1"  # Default discharge rate
        self.battery_id = "unknown"  # Default battery ID

        try:
            # Try to parse the temperature (assuming format like "CY25-...")
            if file_name.startswith("CY") and len(file_name) > 4:
                self.temperature = int(file_name[2:4])

            # Try to parse charge/discharge rates and battery ID
            if "-" in file_name and "#" in file_name:
                # Parse C-rates if available in format like "1_1" (charge_rate_discharge_rate)
                c_rate_part = file_name.split('-')[1].split('#')[0]
                if "*" in c_rate_part:
                    c_rates = c_rate_part.split('*')
                    self.charge_c_rate = c_rates[0]
                    if len(c_rates) > 1:
                        self.discharge_c_rate = c_rates[1]
                elif "_" in c_rate_part:
                    c_rates = c_rate_part.split('_')
                    self.charge_c_rate = c_rates[0]
                    if len(c_rates) > 1:
                        self.discharge_c_rate = c_rates[1]

                # Parse battery ID
                if "#" in file_name:
                    self.battery_id = file_name.split('#')[-1].split('.')[0]
        except Exception as e:
            print(f"Warning: Error parsing filename '{file_name}': {e}")
            print("Using default values for some battery parameters.")
        self.cycle_index = self._get_cycle_index()
        self.cycle_life = len(self.cycle_index)
        print('-'*40, f' Battery #{self.battery_id} ', '-'*40)
        print('电池寿命：', self.cycle_life)
        print('实验温度：', self.temperature)
        print('充电倍率：', self.charge_c_rate)
        print('放电倍率：', self.discharge_c_rate)
        print('变量名：', list(self.df.columns))
        print('-'*100)

    def _get_cycle_index(self):
        cycle_num = np.unique(self.df['cycle number'].values)
        return cycle_num

    def _check(self, cycle=None, variable=None):
        '''
        检查输入的cycle和variable是否合法
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: bool: 是否合法
        '''
        if cycle is not None:
            if cycle not in self.cycle_index:
                raise ValueError('cycle should be in [{},{}]'.format(int(self.cycle_index.min()), int(self.cycle_index.max())))
        if variable is not None:
            if variable not in self.df.columns:
                raise ValueError('variable should be in {}'.format(list(self.df.columns)))
        return True

    def get_cycle(self, cycle):
        '''
        获取第cycle次循环的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的数据
        '''
        self._check(cycle=cycle)
        cycle_df = self.df[self.df['cycle number'] == cycle]
        return cycle_df

    def get_degradation_trajectory(self):
        '''
        获取电池的容量退化轨迹 (mAh)
        :return: list: 容量退化轨迹 (mAh)
        '''
        capacity = []
        for cycle in self.cycle_index:
            cycle_df = self.get_cycle(cycle)
            capacity.append(cycle_df['Q discharge/mA.h'].max())
        return capacity

    def get_value(self, cycle, variable):
        '''
        获取第cycle次循环的variable变量的值
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: series: 第cycle次循环的variable变量的值
        '''
        self._check(cycle=cycle, variable=variable)
        cycle_df = self.get_cycle(cycle)
        return cycle_df[variable].values

    def get_charge_stage(self, cycle):
        '''
        获取第cycle次循环的CCCV阶段的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的CCCV阶段的数据
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        charge_df = cycle_df[cycle_df['control/V/mA'] > 0]
        return charge_df

    def get_CC_stage(self, cycle, voltage_range=None):
        '''
        获取第cycle次循环的CC阶段的数据
        :param cycle: int: 循环次数
        :param voltage_range: list: 电压范围
        :return: DataFrame: 第cycle次循环的CC阶段的数据
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        CC_df = cycle_df[cycle_df['control/mA'] > 0]

        if voltage_range is not None:
            CC_df = CC_df[CC_df['Ecell/V'].between(voltage_range[0], voltage_range[1])]
        return CC_df

    def get_CV_stage(self, cycle, current_range=None):
        '''
        获取第cycle次循环的CV阶段的数据
        :param cycle: int: 循环次数
        :param current_range: list: 电流范围
        :return: DataFrame: 第cycle次循环的CV阶段的数据
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        CV_df = cycle_df[cycle_df['control/V'] > 0]

        if current_range is not None:
            CV_df = CV_df[CV_df['<I>/mA'].between(np.min(current_range), np.max(current_range))]
        return CV_df

    def extract_features(self, cycle):
        '''
        提取第cycle次循环的16个特征
        :param cycle: int: 循环次数
        :return: dict: 16个特征的字典
        '''
        features = {}

        # 获取CC和CV阶段数据
        CC_df = self.get_CC_stage(cycle)
        CV_df = self.get_CV_stage(cycle)
        cycle_df = self.get_cycle(cycle)
        charge_df = self.get_charge_stage(cycle)

        # 检查是否有数据
        if CC_df.empty or CV_df.empty:
            print(f"警告: Cycle {cycle} 的CC或CV阶段数据为空")
            return None

        # 1. CC_time: 电池的恒流充电时间
        cc_time = CC_df['time/s'].max() - CC_df['time/s'].min() if not CC_df.empty else 0
        features['CC_time'] = np.clip(cc_time, 0, 1e6)  # 添加上限防止异常值

        # 2. CV_time: 电池的恒压充电时间
        cv_time = CV_df['time/s'].max() - CV_df['time/s'].min() if not CV_df.empty else 0
        features['CV_time'] = np.clip(cv_time, 0, 1e6)  # 添加上限防止异常值

        # 3. CC_Q: 电池的恒流充电电量 (mAh)
        if not CC_df.empty and 'Q charge/mA.h' in CC_df.columns:
            cc_q = CC_df['Q charge/mA.h'].max() - CC_df['Q charge/mA.h'].min()
        else:
            cc_q = 0
        features['CC_Q'] = np.clip(cc_q, 0, 100000)  # 设置合理上限

        # 4. CV_Q: 电池的恒压充电电量 (mAh)
        if not CV_df.empty and 'Q charge/mA.h' in CV_df.columns:
            cv_q = CV_df['Q charge/mA.h'].max() - CV_df['Q charge/mA.h'].min()
        else:
            cv_q = 0
        features['CV_Q'] = np.clip(cv_q, 0, 100000)  # 设置合理上限

        # 获取充电阶段的电压和电流数据用于计算特征
        if not charge_df.empty:
            voltage = charge_df['Ecell/V'].values
            current = charge_df['<I>/mA'].values
            time = charge_df['time/s'].values

            # 去除异常值并确保所有数组长度一致
            voltage = self._remove_outliers(voltage)
            current = self._remove_outliers(current)

            # 关键修复: 确保所有数组长度一致
            # 找出三个数组中最短的长度
            min_length = min(len(voltage), len(current), len(time))

            # 截断所有数组到相同长度
            voltage = voltage[:min_length]
            current = current[:min_length]
            time = time[:min_length]

            # 检查数据
            if min_length < 5:
                print(f"警告: Cycle {cycle} 的电压或电流数据点不足")
                voltage_slope = 0
                voltage_entropy = 0
                voltage_kurtosis = 0
                voltage_skewness = 0
                current_slope = 0
                current_entropy = 0
                ir = 0
            else:
                # 5. voltage_slope: 电池的电压斜率
                try:
                    time_reshaped = time.reshape(-1, 1)
                    model = LinearRegression().fit(time_reshaped, voltage)
                    voltage_slope = model.coef_[0]
                    # 限制斜率值防止异常
                    voltage_slope = np.clip(voltage_slope, -1, 1)
                except Exception as e:
                    print(f"计算电压斜率时出错: {e}")
                    voltage_slope = 0
                features['voltage_slope'] = voltage_slope

                # 6. voltage_entropy: 电池的电压熵
                # 将电压数据分成多个bins来计算熵
                try:
                    # 确保电压数据有足够的变化范围
                    if np.max(voltage) - np.min(voltage) > 1e-6:
                        hist, _ = np.histogram(voltage, bins=min(20, len(voltage) // 5 + 1), density=True)
                        # 添加小值防止log(0)，使用更安全的计算方法
                        hist = hist + 1e-10
                        hist = hist / np.sum(hist)  # 重新归一化
                        voltage_entropy = entropy(hist)
                    else:
                        voltage_entropy = 0
                except Exception as e:
                    print(f"计算电压熵时出错: {e}")
                    voltage_entropy = 0
                features['voltage_entropy'] = np.clip(voltage_entropy, 0, 10)  # 限制合理范围

                # 7. voltage_kurtosis: 电池的电压峰度
                try:
                    if len(voltage) > 10:  # 确保有足够的数据点
                        voltage_kurtosis = kurtosis(voltage, fisher=True)  # Fisher峰度，均值为0
                    else:
                        voltage_kurtosis = 0
                except Exception as e:
                    print(f"计算电压峰度时出错: {e}")
                    voltage_kurtosis = 0
                features['voltage_kurtosis'] = np.clip(voltage_kurtosis, -5, 5)  # 限制合理范围

                # 8. voltage_skewness: 电池的电压偏度
                try:
                    if len(voltage) > 10:
                        voltage_skewness = skew(voltage)
                    else:
                        voltage_skewness = 0
                except Exception as e:
                    print(f"计算电压偏度时出错: {e}")
                    voltage_skewness = 0
                features['voltage_skewness'] = np.clip(voltage_skewness, -5, 5)  # 限制合理范围

                # 9. current_slope: 电池的电流斜率
                try:
                    time_reshaped = time.reshape(-1, 1)
                    model = LinearRegression().fit(time_reshaped, current)
                    current_slope = model.coef_[0]
                    # 限制斜率值防止异常
                    current_slope = np.clip(current_slope, -100, 100)
                except Exception as e:
                    print(f"计算电流斜率时出错: {e}")
                    current_slope = 0
                features['current_slope'] = current_slope

                # 10. current_entropy: 电池的电流熵
                try:
                    # 确保电流数据有足够的变化范围
                    if np.max(current) - np.min(current) > 1e-6:
                        hist, _ = np.histogram(current, bins=min(20, len(current) // 5 + 1), density=True)
                        # 添加小值防止log(0)，使用更安全的计算方法
                        hist = hist + 1e-10
                        hist = hist / np.sum(hist)  # 重新归一化
                        current_entropy = entropy(hist)
                    else:
                        current_entropy = 0
                except Exception as e:
                    print(f"计算电流熵时出错: {e}")
                    current_entropy = 0
                features['current_entropy'] = np.clip(current_entropy, 0, 10)  # 限制合理范围

                # 11. IR: 根据电流和电压计算出的内阻
                try:
                    # 计算差分
                    current_diff = np.diff(current)
                    voltage_diff = np.diff(voltage)

                    # 确保差分后的数组长度一致
                    min_diff_length = min(len(current_diff), len(voltage_diff))
                    current_diff = current_diff[:min_diff_length]
                    voltage_diff = voltage_diff[:min_diff_length]

                    # 防止除零和异常值
                    mask = (np.abs(current_diff) > 1e-3) & (np.isfinite(current_diff)) & (np.isfinite(voltage_diff))

                    if np.sum(mask) > 5:  # 确保有足够的有效数据点
                        # 计算每个点的内阻，然后取中位数以减少离群值影响
                        point_ir = voltage_diff[mask] / (current_diff[mask] + 1e-6)
                        ir = np.median(np.abs(point_ir))
                    else:
                        # 使用更稳健的方法
                        valid_idx = np.where((np.abs(current) > 1e-3) & np.isfinite(current) & np.isfinite(voltage))[0]
                        if len(valid_idx) > 5:
                            ir = np.std(voltage[valid_idx]) / (np.std(current[valid_idx]) + 1e-6)
                        else:
                            ir = 0
                except Exception as e:
                    print(f"计算内阻时出错: {e}")
                    ir = 0
                # 限制内阻值在合理范围内
                features['IR'] = np.clip(ir, 0, 10)  # 假设内阻不应超过10欧姆
        else:
            # 如果充电数据为空，设置默认值
            features['voltage_slope'] = 0
            features['voltage_entropy'] = 0
            features['voltage_kurtosis'] = 0
            features['voltage_skewness'] = 0
            features['current_slope'] = 0
            features['current_entropy'] = 0
            features['IR'] = 0

        # 12. QC: 电池充电过程中的充电量 (mAh)
        if not charge_df.empty and 'Q charge/mA.h' in charge_df.columns:
            qc = charge_df['Q charge/mA.h'].max() - charge_df['Q charge/mA.h'].min()
        else:
            qc = 0
        features['QC'] = np.clip(qc, 0, 100000)  # 设置合理上限

        # 13-15. TA, TM, TX: 电池的平均、最低和最高温度
        if not charge_df.empty and 'Temperature/℃' in charge_df.columns:
            temp = charge_df['Temperature/℃'].values
            # 过滤异常温度值
            valid_temp = temp[(temp > 0) & (temp < 100) & np.isfinite(temp)]
            if len(valid_temp) > 0:
                features['TA'] = np.mean(valid_temp)
                features['TM'] = np.min(valid_temp)
                features['TX'] = np.max(valid_temp)
            else:
                # 使用环境温度作为近似值
                features['TA'] = self.temperature
                features['TM'] = self.temperature
                features['TX'] = self.temperature
        else:
            # 使用环境温度作为近似值
            features['TA'] = self.temperature
            features['TM'] = self.temperature
            features['TX'] = self.temperature

        # 16. CT: 电池的充电时间
        if not charge_df.empty:
            ct = charge_df['time/s'].max() - charge_df['time/s'].min()
        else:
            ct = 0
        features['CT'] = np.clip(ct, 0, 1e6)  # 添加上限防止异常值

        # 17. capacity: 电池放电过程中传输的电量（放电量）(mAh)
        if not cycle_df.empty and 'Q discharge/mA.h' in cycle_df.columns:
            capacity = cycle_df['Q discharge/mA.h'].max()
        else:
            capacity = 0
        features['capacity'] = np.clip(capacity, 0, 100000)  # 设置合理上限

        # 最后检查所有特征，确保没有nan或inf
        for key in features:
            value = features[key]
            if not np.isfinite(value) or np.isnan(value):
                print(f"警告: 特征 {key} 的值 {value} 无效，替换为0")
                features[key] = 0

        return features

    def _remove_outliers(self, data, threshold=3):
        """
        移除离群值
        :param data: numpy数组
        :param threshold: 标准差倍数阈值
        :return: 处理后的数组
        """
        if len(data) < 3:
            return data

        data = np.array(data)
        # 过滤无限值和NaN
        data = data[np.isfinite(data)]
        if len(data) == 0:
            return np.array([])

        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:  # 几乎无变化的数据
            return data

        # 使用Z-score方法去除离群值
        z_scores = np.abs((data - mean) / std)
        return data[z_scores < threshold]

    def extract_all_features(self):
        '''
        提取所有循环的特征
        :return: DataFrame: 所有循环的特征
        '''
        all_features = []
        for cycle in self.cycle_index:
            features = self.extract_features(cycle)
            if features is not None:
                features['cycle'] = cycle
                all_features.append(features)

        # 转换为DataFrame
        features_df = pd.DataFrame(all_features)

        # 处理可能的缺失值
        if not features_df.empty:
            features_df = features_df.fillna(0)  # 填充缺失值

            # 检查每一列，移除极端值
            for col in features_df.columns:
                if col == 'cycle':
                    continue
                # 使用四分位数范围 (IQR) 方法检测并替换极端值
                Q1 = features_df[col].quantile(0.25)
                Q3 = features_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                # 替换极端值为边界值
                features_df.loc[features_df[col] < lower_bound, col] = lower_bound
                features_df.loc[features_df[col] > upper_bound, col] = upper_bound

        return features_df

    def save_features(self, output_dir='./features'):
        '''
        保存特征到CSV文件
        :param output_dir: str: 输出目录
        :return: str: 保存的文件路径
        '''
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 提取特征
        features_df = self.extract_all_features()

        # 使用原始文件名（不带路径和扩展名）作为保存文件名的一部分
        original_name = os.path.splitext(self.original_file_name)[0]
        output_file = os.path.join(output_dir, f'{original_name}.csv')

        # 检查是否有数据
        if features_df.empty:
            print(f"警告: 未能提取任何有效特征，无法保存文件 {output_file}")
            return None

        # 去掉 cycle 列，保存到 CSV
        if 'cycle' in features_df.columns:
            save_df = features_df.drop('cycle', axis=1)
        else:
            save_df = features_df.copy()

        # 最后检查是否有无效值
        if save_df.isnull().any().any() or (save_df.abs() > 1e10).any().any():
            print("警告: 数据中存在无效值，将被替换为0")
            save_df = save_df.fillna(0)
            save_df = save_df.clip(-1e10, 1e10)  # 限制极值

        save_df.to_csv(output_file, index=False)

        print(f"特征已保存到: {output_file}")
        print(f"提取的特征总数: {len(save_df)}")
        print(f"特征列: {list(save_df.columns)}")

        return output_file


def process_dataset(dataset_path, output_base_dir):
    """
    处理指定数据集目录下的所有CSV文件
    :param dataset_path: 数据集路径
    :param output_base_dir: 输出基础目录
    """
    # 确保路径存在
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return

    # 创建与数据集同名的输出目录
    dataset_name = os.path.basename(dataset_path)
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始处理数据集: {dataset_name}")

    # 查找数据集目录下的所有CSV文件
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

    if not csv_files:
        print(f"在 {dataset_path} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        try:
            print(f"[{i+1}/{len(csv_files)}] 处理文件: {os.path.basename(csv_file)}")
            battery = Battery(path=csv_file)
            battery.save_features(output_dir=output_dir)
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    print(f"数据集 {dataset_name} 处理完成")


def process_all_datasets(base_path):
    """
    处理指定基础路径下的所有数据集
    :param base_path: 数据集基础路径
    """
    # 要处理的数据集列表
    datasets = [
        "Dataset_1_NCA_battery",
        "Dataset_2_NCM_battery",
        "Dataset_3_NCM_NCA_battery"
    ]

    print(f"开始批量处理数据集，基础路径: {base_path}")

    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        if os.path.exists(dataset_path):
            process_dataset(dataset_path, base_path)
        else:
            print(f"数据集路径不存在: {dataset_path}")

    print("所有数据集处理完成")


if __name__ == '__main__':
    # 基础路径，根据实际情况修改
    base_path = "../MyTJUdata"

    # 处理所有数据集
    process_all_datasets(base_path)

    print("特征提取完成！")