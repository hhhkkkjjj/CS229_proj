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


def main(profile, input_col, label_col, ransac, cross, profile_test):
    """
    :param profile: read profile_id, int type
    :param input_col: list of X name
    :param label_col: list of y name
    :param cross: whether it is a cross test between two profiles
    :param profile_test: the test profile if cross == True
    :return: n/a
    """
    torch.manual_seed(2)
    np.random.seed(2)

    path = 'profile_data/Profile_' + str(profile) + '.csv'
    if not cross:
        x, y = util.load_dataset(path, input_col, label_col)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=5)
    else:
        x_train, y_train = util.load_dataset(path, input_col, label_col)

    n_label = len(label_col) # number of kinds of labels
    n_input = len(input_col) # number of kinds of outputs

    # initialization for theta
    theta = np.zeros((n_label, n_input + 1))

    if cross:
        # test path
        path = 'profile_data/Profile_' + str(profile_test) + '.csv'
        x_test, y_test = util.load_dataset(path, input_col, label_col)

    batch_size = 10

    pm_train_dataset = PMSMDataset(x_train, y_train)
    pm_train_loader = torch.utils.data.dataloader.DataLoader(pm_train_dataset, batch_size= batch_size)

    pm_test_dataset = PMSMDataset(x_test, y_test)
    pm_test_loader = torch.utils.data.dataloader.DataLoader(pm_test_dataset, batch_size= 1)
    for i in range(n_label):
        print('For label', label_col[i], " : ")
        ## K-nearest neighbors
        # kNeighbor(x_train, y_train[:, i], x_test, y_test[:, i])
        ## Neural Network
        net = Network(x_train.shape[0], n_input).double()
        lr = 0.001

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        training_losses = []
        for epoch in range(50):
            running_loss = 0.0
            batch_losses = []
            print(pm_train_loader)
            for i, (data, target) in enumerate(pm_train_loader):

                optimizer.zero_grad()

                out = net(data)

                loss = criterion(out, target)
                batch_losses.append(loss.item())

                loss.backward()
                optimizer.step()
            training_losses.append(np.mean(batch_losses))
            print("Epoch {}, loss {:.6f}".format(epoch+1, training_losses[-1]))
        # score = reg.score(x_train, y_train[:, i])
    #     scores = cross_val_score(reg, x_test, y_test[:, i], cv=5, scoring='neg_mean_squared_error')
    #     score = -scores.mean()
    #     if score > high_score:
    #         high_score = score
    #         label = label_col[i]
    #     x_new = util.add_intercept(x_test)
    #     pred = x_new.dot(theta[i, :])
    #     den = pd.DataFrame({'Actual': y_test[:, i],
    #                         'Prediction': pred, })
    #     p = den.plot.kde()
    #     fig = p.get_figure()
    #     fig.savefig("density_plot/" + label_col[i] + '_density.png')
    #     print('Multiple Linear Regression MSE for', label_col[i], 'is',
    #           score, '+-', scores.std()*2)
    # print('In profile', profile, 'the highest score for', label, 'is',high_score)

    # if only one input, we can plot it
    # if n_input == 1:
    #     for i in range(n_label):
    #         save_path = "plots/" + \
    #                     input_col[0] + '_vs_' + label_col[i] + '.png'
    #         util.plot(x_test, y_test[:, i], theta[i, :],
    #                   input_col[0], label_col[0], save_path)
    # print("theta is: ")
    # print(theta)

def kNeighbor(X_train, Y_train, X_test, Y_test):
    from sklearn.neighbors import KNeighborsRegressor
    import math
    rmse_val = []  # to store rmse values for different k
    for K in range(20):
        K = K + 1
        model = KNeighborsRegressor(n_neighbors=K,weights='distance')

        model.fit(X_train, Y_train)  # fit the model
        pred = model.predict(X_test)  # make prediction on test set
        error = math.sqrt(mean_squared_error(Y_test, pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
        print('RMSE value for k= ', K, 'is:', error)

if __name__ == '__main__':
    # input_col = ["ambient", "coolant", "motor_speed", "i_d", "i", "u","Time"]
    input_col = ["ambient", "coolant", "i", "u","Time"]
    label_col = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
    path = 'profile_data/Profile_4.csv'
    main(profile=4,
        input_col=input_col,
        label_col=label_col,
        ransac=True,
        cross = False,  # cross test: True
        profile_test=6) # if cross test, profile used for test
