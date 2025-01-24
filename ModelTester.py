import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from DataPre import Data_Pre
from AttentionWithNoDeconvPos import AttentionWithNoDeconvPos


class ModelTester:
    def __init__(self, data_path, model_path, sub_dir, d_model, n_head, device='cpu'):
        self.data_path = data_path
        self.model_path = model_path
        self.device = torch.device(device)
        self.sub_dir = sub_dir
        self.d_model = d_model
        self.n_head = n_head
        self.data_x, self.data_y = self.load_data()
        self.model = self.load_model()

    def load_data(self):
        return Data_Pre(self.data_path)

    def load_model(self):
        model = AttentionWithNoDeconvPos(input_size=self.data_x.shape[1],
                                         d_model=self.d_model,
                                         n_head=self.n_head,
                                         max_len=self.data_x.shape[1],
                                         num_neurons=self.data_x.shape[1],
                                         device=self.device,
                                         hidden=self.data_y.shape[2] ** 2)
        path = os.path.join(self.model_path,f'{self.sub_dir}_cluster_model.pth')
        # print(path)
        # print(type(torch.load(path, map_location=self.device)))
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            # model = torch.load(path, map_location=self.device)
        else:
            raise FileNotFoundError(f"No model found at {path}")

        model.to(self.device)
        model.eval()
        return model

    def evaluate(self):
        input = torch.FloatTensor(self.data_x).to(self.device)
        output = torch.FloatTensor(self.data_y).to(self.device)
        pred, atten, attention_input = self.model(input)
        # Squeeze to remove any extraneous dimensions if necessary
        output_true = output.data.cpu().numpy()
        output_true = output_true.squeeze()
        output_true = output_true.astype(dtype='float64')
        output_pre = pred.data.cpu().numpy()
        output_pre = output_pre.squeeze()
        output_pre = output_pre.astype(dtype='float64')
        attention_map = atten.data.cpu().numpy()
        attention_map = attention_map.squeeze()
        attention_map = attention_map.astype(dtype='float64')
        r2_values = [r2_score(true_i.flatten(), pred_i.flatten()) for true_i, pred_i in zip(output_true, output_pre)]
        mean_r2 = np.mean(r2_values)
        return r2_values, mean_r2, attention_map, output_true, output_pre

    def plot_r2_values(self, r2_values, mean_r2):
        plt.plot(range(1, len(r2_values) + 1), r2_values, marker='o', linestyle='-')
        plt.xlabel('Trigger conditions')
        plt.ylabel('R-squared Values')
        plt.title(f'R-squared change // Mean R2 Value: {mean_r2}')
        plt.grid(True)
        plt.show()




