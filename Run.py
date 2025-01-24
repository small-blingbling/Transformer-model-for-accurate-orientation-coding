## configuration
import matplotlib.pyplot as plt
import torch
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置随机参数：保证实验结果可以重复
SEED = 1234

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 参数设置
epochs = 1000
batch_size = 2 # 大于2,但是我这边sample比较小，所以大了容易过拟合
learning_rate = 0.00005 ## 我试了一下，就是学习率小点效果更好，慢慢学
test_size = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
d_model = 2
n_head = 1
alpha = 0.9

from DataPre import Data_Pre
from ModelTrainer import ModelTrainer
from ModelTester import ModelTester
import os
from scipy.io import savemat

def main():
    start_time = time.time()
    current_file_path = __file__
    top_directory = os.path.dirname(current_file_path)
    # sub_dirs = ['MA','MB','MC','MD','ME','MH','MI','MJ']
    # sub_dirs = ['MB','MC','MD','ME','MH','MI','MJ']
    sub_dirs = ['Demo']
    model_dir = 'Model'
    model_path = os.path.join(top_directory, model_dir)
    for sub_dir in sub_dirs:
        data_path = os.path.join(top_directory, sub_dir, 'Data')
        data_x, data_y = Data_Pre(data_path)
        # print(data_x.shape)
        print(f'{sub_dir}_{data_x.shape[1]} neurons')

        trainer = ModelTrainer( data_x, data_y, device = device,
            epochs = epochs, test_size = test_size, batch_size = batch_size, learning_rate = learning_rate,
            d_model = d_model, n_head = n_head, seed = SEED, alpha = alpha)

        # 开始训练，并保存训练好的模型
        best_model, train_loss, val_loss = trainer.train()
        model_save_path = os.path.join(model_path, f'{sub_dir}_cluster_model.pth')
        torch.save(best_model.state_dict(), model_save_path)

        loss_save_path = os.path.join(top_directory, sub_dir)
        savemat(os.path.join(loss_save_path, 'loss_data.mat'), {'train_loss': train_loss, 'val_loss': val_loss})
        print(data_path)
        print(model_path)
        tester = ModelTester(data_path = data_path, model_path = model_path, sub_dir = sub_dir, d_model= d_model ,n_head = n_head,device = device)
        r2_values, mean_r2, attention_map, output_true, output_pred = tester.evaluate()
        print(r2_values)
        print(mean_r2)
        # tester.plot_r2_values(r2_values, mean_r2)
        # 我这里是存的r2，但是实际上分析的时候用的ssim这个指标
        savemat(os.path.join(loss_save_path, 'test_data.mat'), {'r2_values': r2_values, 'mean_r2': mean_r2})
        savemat(os.path.join(loss_save_path, 'attention_map.mat'), {'attention_map': attention_map})
        savemat(os.path.join(loss_save_path, 'output_true.mat'), {'output_true': output_true})
        savemat(os.path.join(loss_save_path, 'output_pred.mat'), {'output_pred': output_pred})
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
if __name__ == '__main__':
    main()

