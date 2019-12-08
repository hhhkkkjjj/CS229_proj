import numpy as np
import util
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from sklearn.preprocessing import MinMaxScaler

class PMSMDataset(torch.utils.data.dataset.Dataset):
    """Dataset with Rotor Temperature as Target"""
    def __init__(self, data, target):

        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.target[idx]

class Network(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(Network, self).__init__()


        self.conv1 = nn.Conv1d(1, 3, kernel_size=(sequence_length, n_features))

        self.lin_in_size = self.conv1.out_channels * int(((sequence_length - (self.conv1.kernel_size[0]-1) -1)/self.conv1.stride[0] +1))

#         print(self.lin_in_size)

        self.fc1 = nn.Linear(self.lin_in_size,30)
        self.fc2 = nn.Linear(30, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = x.view(-1, self.lin_in_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def build_sequences(features_df, target_df, sequence_length = 10):
    """Builds sequences from data and converts them into pytorch tensors
        sequence_length - represents the number of samples to be considered in a sequence
    """
    data_ = []
    target_ = []

    for i in range(int(features_df.shape[0]/sequence_length)):

        data = torch.from_numpy(features_df.iloc[i:i+sequence_length].values)
        #data=torch.from_numpy(features_df[i:i+sequence_length])
        target = torch.from_numpy(target_df.iloc[i+sequence_length+1].values)
        #target=torch.from_numpy(target_df[i+sequence_length+1])

        data_.append(data)
        target_.append(target)

    data = torch.stack(data_)
    target = torch.stack(target_)

    return data, target

def main():
    df = pd.read_csv("pmsm_temperature_data.csv")
    col_list = df.columns.tolist()
    profile_id = ['profile_id']
    target_list = ['pm', 'torque', 'stator_yoke', 'stator_tooth', 'stator_winding']
    feature_list = [col for col in col_list if col not in target_list and col not in profile_id]

    col_list = df.columns.tolist()
    profile_id = ['profile_id']
    target_list = ['pm', 'torque', 'stator_yoke', 'stator_tooth', 'stator_winding']
    feature_list = [col for col in col_list if col not in target_list and col not in profile_id]

    df['profile_id'] = df.profile_id.astype('category', inplace=True)
    df_dict = {}
    for id_ in df.profile_id.unique():
        df_dict[id_] = df[df['profile_id']==id_].reset_index(drop = True)

    prof_id = 4

    curr_df = df_dict[prof_id]

    curr_df = curr_df.drop('profile_id', axis = 1)
    columns = curr_df.columns.tolist()

    scaler = MinMaxScaler()

    curr_df = pd.DataFrame(scaler.fit_transform(curr_df), columns= columns)
    sequence_length = 3

    features = curr_df[feature_list]
    target = curr_df[target_list][['pm']]

    data, target = build_sequences(features, target, sequence_length=sequence_length)

    test_size = 0.05

    indices = torch.randperm(data.shape[0])

    train_indices = indices[:int(indices.shape[0] * (1-test_size))]
    test_indices = indices[int(indices.shape[0] * (1-test_size)):]

    X_train, y_train = data[train_indices], target[train_indices]
    X_test, y_test = data[test_indices], target[test_indices]

    batch_size = 10

    pm_train_dataset = PMSMDataset(X_train, y_train)
    pm_train_loader = torch.utils.data.dataloader.DataLoader(pm_train_dataset, batch_size= batch_size)

    pm_test_dataset = PMSMDataset(X_test, y_test)
    pm_test_loader = torch.utils.data.dataloader.DataLoader(pm_test_dataset, batch_size= 1)

    n_features = X_train.shape[-1]

    net = Network(sequence_length, n_features).double()

    lr = 0.001

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    training_losses = []
    for epoch in range(50):
        running_loss = 0.0
        batch_losses = []
        for i, (data, target) in enumerate(pm_train_loader):

            optimizer.zero_grad()

            out = net(data)

            loss = criterion(out, target)
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()
        training_losses.append(np.mean(batch_losses))
        print("Epoch {}, loss {:.6f}".format(epoch+1, training_losses[-1]))


if __name__ == '__main__':
    # input_col = ["ambient", "coolant", "motor_speed", "i_d", "i", "u","Time"]
    input_col = ["ambient", "coolant", "i", "u","Time"]
    label_col = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
    path = 'profile_data/Profile_4.csv'
    main() # if cross test, profile used for test
