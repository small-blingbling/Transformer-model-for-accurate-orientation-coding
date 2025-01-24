import numpy as np

from AttentionWithNoDeconvPos import AttentionWithNoDeconvPos
import torch
import gc
import os
from scipy.io import savemat
from DataPre import Data_Pre
import time

start_time = time.time()
current_file_path = os.getcwd()
top_directory = os.path.dirname(current_file_path)
print(top_directory)
# sub_dirs = ['MA','MB','MC','MD','ME','MH','MI','MJ']
sub_dirs = ['Demo']
model_dir = 'Model'
model_path = os.path.join(top_directory, model_dir)
def matlab_range(start, stop):
    return list(range(start-1, stop))
Ori_range = np.array([
    # matlab_range(97, 108),
    # matlab_range(109, 120),
    # matlab_range(37, 48),
    # matlab_range(61, 72),
     matlab_range(49, 60),
    # matlab_range(1, 12),
    # matlab_range(73, 84),
    # matlab_range(109, 120),
# matlab_range(37, 48)
])
print(Ori_range)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for sub_dir in sub_dirs:
    data_path = os.path.join(top_directory, sub_dir, 'Data')
    data_x, data_y = Data_Pre(data_path)
    print(data_path)
    print(f'{sub_dir}_{data_x.shape[1]} neurons')
    indices = Ori_range[sub_dirs.index(sub_dir), :]  # 获取Ori_range中相应子目录的行索引
    print(indices)
    input = torch.FloatTensor(data_x[indices]).to(device)
    output = torch.FloatTensor(data_y[indices]).to(device)



    input_size = input.shape[1]
    d_model = 2
    n_head = 1
    max_len = input.shape[1]
    drop_prob = 0
    device = device
    hidden = output.shape[2] ** 2
    num_neurons = input.shape[1]

    thresholds = 10 ** np.arange(-1, -3.2, -0.05)
    print(thresholds)
    comparison_operations = [True, False]

    model = AttentionWithNoDeconvPos(input_size=input_size, d_model=d_model, n_head=n_head, max_len=max_len,
                                     num_neurons=num_neurons, device=device, hidden=hidden)
    path = os.path.join(model_path, f'{sub_dir}_cluster_model.pth')

    # if sub_dir == 'MA':
    #     model_path = os.path.join(data_path, 'model.pth')  # 假设模型文件名是 'model.pth'
    # model = torch.load(f'{sub_dir}_cluster_model.pth')
    #     # print(f'{sub_dir}_model_cluster_sub.pth')
    # else:
    # model.load_state_dict(torch.load(f'{sub_dir}_cluster_model.pth'))

    state_dict = torch.load(f'{sub_dir}_cluster_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.to(torch.device('cpu'))

    # 继续进行推理或训练
    model.eval()


    new_batch_size = 2
    num_batches = input.size(0) // new_batch_size

    results = torch.empty((len(thresholds), 2, num_batches * new_batch_size, 79, 79)).to(device)

    for i, threshold in enumerate(thresholds):
        for j, is_greater in enumerate([True, False]):
            for batch_idx in range(num_batches):
                input_batch = input[batch_idx * new_batch_size: (batch_idx + 1) * new_batch_size]
                pre, _, _ = model(input_batch, threshold, is_greater)
                results[i, j, batch_idx * new_batch_size: (batch_idx + 1) * new_batch_size] = pre.squeeze()

                del input_batch, pre
                torch.cuda.empty_cache()
                gc.collect()

    # 不能整除时
    if input.size(0) % new_batch_size != 0:
        input_batch = input[num_batches * new_batch_size:]
        for i, threshold in enumerate(thresholds):
            for j, is_greater in enumerate([True, False]):
                pre, atten, _ = model(input_batch, threshold, is_greater)
                results[i, j, num_batches * new_batch_size:] = pre.squeeze()

    print(results.shape)
    results = results.data.cpu().numpy()
    results = results.astype(np.float64)

    results_save_path = os.path.join(top_directory, sub_dir)
    print(results_save_path)
    savemat(os.path.join(results_save_path, 'results.mat'), {'results': results})

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total runtime: {elapsed_time:.2f} seconds")