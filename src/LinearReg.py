import numpy as np
import pandas as pd
import os
import seaborn as sns
import util
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main(path, input_col, label_col, data_start, data_end):
    """
    Args:
        train_path: Path to CSV file containing dataset for training.
    """
    x, y = util.load_dataset(path, input_col, label_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x[data_start: data_end], y[data_start: data_end], test_size=0.33, random_state=0)
    n_label = len(label_col)
    n_input = len(input_col)
    theta = np.zeros((n_label, n_input + 1))
    for i in range(n_label):
        reg = LinearRegression().fit(x_train, y_train[:, i])
        theta[i, 0] = reg.intercept_
        print(reg.coef_)
        theta[i, 1:] = reg.coef_

        print('Multiple Linear Regression Score : ',
              reg.score(x_test, y_test[:, i]))

    if n_input == 1:
        for i in range(n_label):
            save_path = "plots/" + \
                input_col[0] + '_vs_' + label_col[i] + '.png'
            util.plot(x_test, y_test[:, i], theta[i, :],
                      input_col[0], label_col[0], save_path)
    print("theta is: ")
    print(theta)


if __name__ == '__main__':
    input_col = [ "coolant", "ambient", "i_d", "u_d","motor_speed"]
    label_col = ["pm","stator_yoke", "stator_tooth", "stator_winding"]
    main(path='pmsm_temperature_data.csv',
         input_col=input_col,
         label_col=label_col,
         data_start=0,
         data_end=1000)
