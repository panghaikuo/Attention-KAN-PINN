'''
TJU数据集的结果分析

English:
    This file is used to analyze the results of the TJU dataset.
'''
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class Results:
    def __init__(self,root='../results of PINN/',gap=0.07):
        self.root = root
        self.experiments = os.listdir(root)
        self.gap = gap
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self,train_batch=0,test_batch=1,experiment=1):
        subfolder = f'{train_batch}-{test_batch}/Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder,'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder,'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder,'true_label.npy')

    def parser_log(self):
        '''
        解析train过程中产生的log文件，获取里面的数据
        English:
            Parse the log file generated during the training process to obtain the data
        :return: dict
        '''
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
                data_dict[k] = v

        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        # ... 省略不变更代码 ...
        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                if 'data loss:' in line:  # 检查'data loss:'是否存在于行中
                    train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))
                    train_PDE_loss.append(float(line.split('PDE loss:')[1].split(',')[0]))
                    train_phy_loss.append(float(line.split('physics loss:')[1].split(',')[0]))
                    train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Train]' in line:
                if 'data loss:' in line:  # 检查'data loss:'是否存在于行中
                    train_data_loss.append(float(line.split('data loss:')[1].split(',')[0]))
                    train_PDE_loss.append(float(line.split('PDE loss:')[1].split(',')[0]))
                    train_phy_loss.append(float(line.split('physics loss:')[1].split(',')[0]))
                    train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            # ... 省略不变更代码 ...
            elif '[Valid]' in line:
                valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))
            elif '[Test]' in line:
                test_mse.append(float(line.split('MSE:')[1].split(',')[0]))
                test_epoch.append(int(lines[i - 1].split('epoch:')[1].split(',')[0]))

        data_dict['train_data_loss'] = train_data_loss
        data_dict['train_PDE_loss'] = train_PDE_loss
        data_dict['train_phy_loss'] = train_phy_loss
        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch


        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace('data/TJU data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace('data/TJU data/', '').replace('.csv', '').replace('\'', '').split(', ')
            for i in range(len(line_list)):
                line_list[i] = line_list[i].split('\\')[-1]
            data_dict['IDs_2'] = line_list
            #print('test ID length:',len(line_list))

        return data_dict

    def parser_label(self):
        '''
        解析预测结果
        English:
            Parse the prediction results
        :return:
        '''
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_label, true_label)
        plt.figure(figsize=(6, 3),dpi=200)
        plt.plot(true_label, label='true label')
        plt.plot(pred_label, label='pred label')
        plt.legend()
        # plt.show()

        # 用来保存每个电池的预测结果
        # To save the prediction results of each battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > self.gap)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            [MAE_i, MAPE_i, MSE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            # print('battery {} MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(i + 1, MAE_i, MAPE_i,
            #                                                                                      MSE_i, RMSE_i, R2_i))
            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            MSE_list.append(MSE_i)
            RMSE_list.append(RMSE_i)
            R2_list.append(R2_i)
        #print('Mean  MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(MAE, MAPE, MSE, RMSE, R2))
        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        results_dict['MSE'] = MSE_list
        results_dict['RMSE'] = RMSE_list
        results_dict['R2'] = R2_list
        return results_dict


    def get_test_results(self,train=0,test=1,e=1):
        '''
        解析训练和测试数据中的电池id
        English:
            Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        try:
            self._update_experiments(train_batch=train,test_batch=test,experiment=e)
            log_dict = self.parser_log()
            results_dict = self.parser_label()

            # Check if 'IDs_2' exists in log_dict
            if 'IDs_2' in log_dict:
                # Make sure the length of channel list matches the length of results
                num_batteries = len(results_dict['MAE'])
                if 'IDs_2' in log_dict and len(log_dict['IDs_2']) < num_batteries:
                    # If there are fewer channel IDs than batteries, pad with default values
                    log_dict['IDs_2'].extend([f'Unknown{i+1}' for i in range(num_batteries - len(log_dict['IDs_2']))])
                elif len(log_dict['IDs_2']) > num_batteries:
                    # If there are more channel IDs than batteries, truncate
                    log_dict['IDs_2'] = log_dict['IDs_2'][:num_batteries]

                results_dict['channel'] = log_dict['IDs_2']
            else:
                # If no channel IDs are available, create default ones
                num_batteries = len(results_dict['MAE'])
                results_dict['channel'] = [f'Battery{i+1}' for i in range(num_batteries)]

            return results_dict
        except Exception as e:
            print(f"Error in get_test_results for train={train}, test={test}, e={e}: {str(e)}")
            # Return empty result with same structure
            return {
                'pred_label': [], 'true_label': [], 'MAE': [], 'MAPE': [],
                'MSE': [], 'RMSE': [], 'R2': [], 'channel': []
            }

    def get_battery_average(self,train_batch=0,test_batch=0):
        '''
        计算每次实验中所有电池的平均值
        English:
            Calculate the average value of all batteries in each experiment
        :param train_batch:
        :param test_batch:
        :return: dataframe，每一行是一个实验中所有电池的平均值 (each row is the average value of all batteries in an experiment)
        '''
        df_mean_values = []
        for i in range(1,11):
            try:
                res = self.get_test_results(train_batch,test_batch,i)

                # Check if the result is valid and has data
                if not res or len(res['MAE']) == 0:
                    print(f"Skipping experiment {i} due to empty results")
                    continue

                # Create a new dictionary with only the metrics for the DataFrame
                metrics_dict = {
                    'MAE': res['MAE'],
                    'MAPE': res['MAPE'],
                    'MSE': res['MSE'],
                    'RMSE': res['RMSE'],
                    'R2': res['R2']
                }

                # Ensure all lists have the same length
                min_length = min(len(v) for v in metrics_dict.values())
                for k in metrics_dict:
                    metrics_dict[k] = metrics_dict[k][:min_length]

                # Now create DataFrame with lists of same length
                df_i = pd.DataFrame(metrics_dict)
                df_mean_values.append(df_i.mean(axis=0).values)
            except Exception as e:
                print(f"Error processing experiment {i}: {str(e)}")
                # Add NaN values for this experiment
                df_mean_values.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan]))

        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values,columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'R2'])
        df_mean.insert(0,'experiment',range(1,len(df_mean_values)+1))
        print(df_mean)
        return df_mean


    def get_experiments_mean(self,train_batch=0,test_batch=0):
        '''
        分别获取每个测试电池在所有实验中的平均值
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe，每一行是一个电池在10次实验中的平均值 (each row is the average value of a battery in 10 experiments)
        '''
        # Dictionary to store results for each battery across experiments
        all_battery_results = {}

        # Store all unique channel IDs across experiments
        all_channels = set()

        # First pass: collect all unique battery channels
        for i in range(1,11):
            try:
                res = self.get_test_results(train_batch,test_batch,i)
                for channel in res['channel']:
                    all_channels.add(channel)
            except Exception as e:
                print(f"Error collecting channels in experiment {i}: {str(e)}")

        # Initialize data storage for each battery
        for channel in all_channels:
            all_battery_results[channel] = {
                'MAE': [], 'MAPE': [], 'MSE': [], 'RMSE': [], 'R2': []
            }

        # Second pass: collect metrics for each battery in each experiment
        for i in range(1,11):
            try:
                res = self.get_test_results(train_batch,test_batch,i)

                # Skip if no results
                if not res or len(res['channel']) == 0:
                    continue

                # For each battery in this experiment
                for j, channel in enumerate(res['channel']):
                    if j < len(res['MAE']):  # Ensure index is valid
                        all_battery_results[channel]['MAE'].append(res['MAE'][j])
                        all_battery_results[channel]['MAPE'].append(res['MAPE'][j])
                        all_battery_results[channel]['MSE'].append(res['MSE'][j])
                        all_battery_results[channel]['RMSE'].append(res['RMSE'][j])
                        all_battery_results[channel]['R2'].append(res['R2'][j])
            except Exception as e:
                print(f"Error processing metrics in experiment {i}: {str(e)}")

        # Calculate mean values for each battery
        data_rows = []
        for channel in all_channels:
            if all(len(v) > 0 for v in all_battery_results[channel].values()):
                row = {
                    'channel': channel,
                    'MAE': np.mean(all_battery_results[channel]['MAE']),
                    'MAPE': np.mean(all_battery_results[channel]['MAPE']),
                    'MSE': np.mean(all_battery_results[channel]['MSE']),
                    'RMSE': np.mean(all_battery_results[channel]['RMSE']),
                    'R2': np.mean(all_battery_results[channel]['R2'])
                }
                data_rows.append(row)

        # Create and return DataFrame
        df_mean = pd.DataFrame(data_rows)

        # Format channel names if needed
        if not df_mean.empty:
            df_mean['channel'] = df_mean['channel'].astype(str)
            df_mean['channel'] = df_mean['channel'].apply(lambda x: x[-9:] if len(x) > 9 else x)

        print(df_mean)
        return df_mean

if __name__ == '__main__':

    root = '../MyTJUbestresults/TJU-Attention-KAN-PINN0-0_1 results/'
    writer = pd.ExcelWriter('../MyTJUbestresults/TJU-Attention-KAN-PINN0-0_1 results.xlsx')
    # batch = 1
    results = Results(root)
    # print(f"\nProcessing batch {batch}:")
    # df_battery_mean = results.get_battery_average(train_batch=batch, test_batch=batch)
    # df_experiment_mean = results.get_experiments_mean(test_batch=batch, train_batch=batch)
    #
    # if not df_battery_mean.empty:
    #     df_battery_mean.to_excel(writer, sheet_name=f'battery_mean_{batch}', index=False)
    # else:
    #     print(f"Skipping writing empty battery_mean_{batch}")
    #
    # if not df_experiment_mean.empty:
    #     df_experiment_mean.to_excel(writer, sheet_name=f'experiment_mean_{batch}', index=False)
    #     print(df_experiment_mean.mean())
    # else:
    #     print(f"Skipping writing empty experiment_mean_{batch}")
    # results = Results(root, gap=0.07)
    # df_battery_mean = results.get_battery_average(train_batch=batch, test_batch=batch)
    # df_experiment_mean = results.get_experiments_mean(test_batch=batch, train_batch=batch)
    # df_battery_mean.to_excel(writer, sheet_name='battery_mean_{}'.format(batch), index=False)
    # df_experiment_mean.to_excel(writer, sheet_name='experiment_mean_{}'.format(batch), index=False)
    # print(df_experiment_mean.mean())
    for batch in range(3):
        try:
            print(f"\nProcessing batch {batch}:")
            df_battery_mean = results.get_battery_average(train_batch=batch,test_batch=batch)
            df_experiment_mean = results.get_experiments_mean(test_batch=batch,train_batch=batch)

            if not df_battery_mean.empty:
                df_battery_mean.to_excel(writer,sheet_name=f'battery_mean_{batch}',index=False)
            else:
                print(f"Skipping writing empty battery_mean_{batch}")

            if not df_experiment_mean.empty:
                df_experiment_mean.to_excel(writer,sheet_name=f'experiment_mean_{batch}',index=False)
                print(df_experiment_mean.mean())
            else:
                print(f"Skipping writing empty experiment_mean_{batch}")
        except Exception as e:
            print(f"Error processing batch {batch}: {str(e)}")

    writer.save()