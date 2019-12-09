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
    def __init__(self, data, target):

        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.target[idx]

class Network(nn.Module):
    def __init__(self, sequence_length, n_features):
        ## Two layers of neural network
        super(Network, self).__init__()

        self.conv1 = nn.Conv1d(1, 5, kernel_size=(sequence_length, n_features))

        self.lin_in_size = self.conv1.out_channels * int(((sequence_length - (self.conv1.kernel_size[0]-1) -1)/self.conv1.stride[0] +1))

        self.fc1 = nn.Linear(self.lin_in_size,100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = x.view(-1, self.lin_in_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def convert_sequences(features_df, target_df, sequence_length):
    ## divide data into time sequences and only predict result after the time sequence
    ## reasonable sequence length should be greater than 2
    input_list = []
    output_list = []

    for i in range(int(features_df.shape[0]/sequence_length)):

        input = torch.from_numpy(features_df.iloc[i:i+sequence_length].values)
        output= torch.from_numpy(target_df.iloc[i+sequence_length+1].values)

        input_list.append(input)
        output_list.append(output)

    data = torch.stack(input_list)
    target = torch.stack(output_list)

    return data, target

def divideData(path,prof_id):
    data= pd.read_csv(path)
    data['profile_id'] = data.profile_id.astype('category', inplace=True)
    dict = {}
    for id in data.profile_id.unique():
        dict[id] = data[data['profile_id']==id].reset_index(drop = True)
    return dict[prof_id]

def main(path, prof_id, target_list, feature_list,sequence_length,batch_size,lr,test=True):
    n_features = len(feature_list)
    curr_data = divideData(path,prof_id)

    curr_data = curr_data.drop('profile_id', axis = 1)
    columns = curr_data.columns.tolist()

    scaler = MinMaxScaler()
    curr_data = pd.DataFrame(scaler.fit_transform(curr_data), columns=columns)

    features = curr_data[feature_list]
    target = curr_data[target_list][['pm']]       ##pm is what we care about the most

    data, target = convert_sequences(features, target, sequence_length)

    ##dividing into test and training set
    test_size = 0.08

    indices = torch.randperm(data.shape[0])

    train_indices = indices[:int(indices.shape[0] * (1-test_size))]
    test_indices = indices[int(indices.shape[0] * (1-test_size)):]

    x_train, y_train = data[train_indices], target[train_indices]
    x_test, y_test = data[test_indices], target[test_indices]


    pm_train_dataset = PMSMDataset(x_train, y_train)
    pm_train_loader = torch.utils.data.dataloader.DataLoader(pm_train_dataset, batch_size= batch_size)

    pm_test_dataset = PMSMDataset(x_test, y_test)
    pm_test_loader = torch.utils.data.dataloader.DataLoader(pm_test_dataset, batch_size= 1)


    net = Network(sequence_length, n_features).double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr)
## training network
    training_losses = []
    delta=0.0
    threshold=1e-6
    prev_loss=0
    for epoch in range(50):
        running_loss = 0.0
        batch_losses = []
        for i, (data, target) in enumerate(pm_train_loader):
            optimizer.zero_grad()
            predicted = net(data)

            train_loss = criterion(predicted, target)
            batch_losses.append(train_loss.item())

            train_loss.backward()
            optimizer.step()
        training_losses.append(np.mean(batch_losses))
        delta=abs(training_losses[-1]-prev_loss)
        # print(delta)
        if delta<threshold:
            break
        prev_loss=training_losses[-1]
        print("Epoch {}, loss {:.6f}".format(epoch+1, training_losses[-1]))
    plot_fig(training_losses)

 ## testing
    if test:
        losses = []
        batch_losses = []
        labels = []
        outputs = []
        with torch.no_grad():
            for i, (data, target) in enumerate(pm_test_loader):
                output = net(data)
                loss = criterion(output, target)

                labels.append(target.item())
                outputs.append(output.item())

                batch_losses.append(loss.item())
            losses.append(np.mean(batch_losses))
        print("Testing loss {:.6f}".format(losses[-1]))
        plot_fig_test(outputs,labels)

def plot_fig(training_losses):
    plt.plot(training_losses)
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.savefig('training_loss_lr_'+str(lr)+'_sequence_length'+str(sequence_length)+' .png')

def plot_fig_test(outputs,labels):
    plt.figure(figsize=(13,7))
    plt.plot(np.arange(len(outputs)),outputs, alpha = 0.8, marker = '.',label = 'predicted' )
    plt.scatter(np.arange(len(labels)),labels, c = 'r', s = 15, label = 'labels')
    plt.legend(loc='best')
    plt.savefig('Testing_result_lr_'+str(lr)+'_sequence_length'+str(sequence_length)+' .png')

if __name__ == '__main__':
    # input_col = ["ambient", "coolant", "motor_speed", "i_d", "i", "u","Time"]
    feature_list = ["ambient", "coolant","motor_speed", "i", "u",]
    target_list = ['pm', 'torque', 'stator_yoke', 'stator_tooth', 'stator_winding']
    path = "pmsm_temperature_data.csv"
    profile_id = 4
    sequence_length = 6
    batch_size = 5
    lr = 0.002
    main(path, profile_id, target_list, feature_list,sequence_length,batch_size,lr,test=True) # if cross test, profile used for test
