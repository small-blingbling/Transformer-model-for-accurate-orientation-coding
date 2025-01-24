import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from AttentionWithNoDeconvPos import AttentionWithNoDeconvPos

class ModelTrainer:
    def __init__(self, data_x, data_y, device='cpu',
                 epochs=2000, test_size=0.014, batch_size=2, learning_rate=0.00005,
                 d_model=2, n_head=1, seed = 1234, alpha = 0.9):
        self.data_x = data_x
        self.data_y = data_y
        self.device = device
        self.epochs = epochs
        self.test_size = test_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.d_model = d_model
        self.n_head = n_head
        self.seed = seed
        self.alpha = alpha
        self.train_dataloader, self.test_dataloader = self.prepare_data_loaders()

    def prepare_data_loaders(self):
        dataset = DataSet(self.data_x, self.data_y)
        train_data, test_data = train_test_split(dataset, test_size=self.test_size, random_state=self.seed)
        train_dataloader = Data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dataloader = Data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("TestDataLoader 的batch个数", test_dataloader.__len__())
        print("TrainDataLoader 的batch个数", train_dataloader.__len__())
        return train_dataloader, test_dataloader

    def eval_test(self, model):
        test_epoch_loss = []
        with torch.no_grad():
            for test_x, test_y in self.test_dataloader:
                test_x = test_x.to(self.device)
                test_y = test_y.to(self.device)
                test_pre, _, _ = model(test_x)
                test_loss = torch.nn.MSELoss()(test_pre, test_y)
                test_epoch_loss.append(test_loss.item())
        return np.mean(test_epoch_loss)

    def train(self):
        model = AttentionWithNoDeconvPos(input_size=self.data_x.shape[1],
            d_model=self.d_model,
            n_head=self.n_head,
            max_len=self.data_x.shape[1],
            num_neurons=self.data_x.shape[1],
            device=self.device,
            hidden=self.data_y.shape[2]**2)
        model.to(self.device)
        params_to_optimize = [
            {"params": model.attention.parameters(), "weight_decay": 0},
            {"params": model.ff1.parameters(), "weight_decay": 0.01},
            {"params": model.pos.parameters(), "weight_decay": 0},
            {"params": model.ff.parameters(), "weight_decay": 0},
            {"params": model.embedding.parameters(), "weight_decay": 0}
        ]
        optimizer = torch.optim.RMSprop(params_to_optimize, lr=self.learning_rate, alpha=self.alpha)

        best_test_loss = float('inf')
        train_loss_history = []
        test_loss_history = []

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = []
            for train_x, train_y in self.train_dataloader:
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                # print(train_x.shape)
                train_pre, _, _ = model(train_x)
                # print(train_pre.shape)
                loss = torch.nn.MSELoss()(train_pre, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            train_epoch_loss = np.mean(epoch_loss)
            test_epoch_loss = self.eval_test(model)

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                print("best_test_loss", best_test_loss)
                best_model = model

            train_loss_history.append(train_epoch_loss)
            test_loss_history.append(test_epoch_loss)
            print(f"Epoch: {epoch}, Train Loss: {train_epoch_loss}, Test Loss: {test_epoch_loss}")

        return best_model, train_loss_history, test_loss_history


class DataSet(Data.Dataset):
    def __init__(self, input, label):
        self.inputs = torch.FloatTensor(input)
        self.labels = torch.FloatTensor(label)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


