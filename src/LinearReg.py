import numpy as np
import util
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd

def main(profile, input_col, label_col, ransac, cross, profile_test):
    """
    :param profile: read profile_id, int type
    :param input_col: list of X name
    :param label_col: list of y name
    :param cross: whether it is a cross test between two profiles
    :param profile_test: the test profile if cross == True
    :return: n/a
    """
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

    high_score=0
    for i in range(n_label):
        print('For label', label_col[i], " : ")
        kNeighbor(x_train, y_train[:, i], x_test, y_test[:, i])
        if not ransac:
            reg = LinearRegression().fit(x_train, y_train[:, i])
            theta[i, 0] = reg.intercept_
            theta[i, 1:] = reg.coef_
        else:
            reg = RANSACRegressor().fit(x_train, y_train[:, i])
            theta[i, 0] = reg.estimator_.intercept_
            theta[i, 1:] = reg.estimator_.coef_

        # score = reg.score(x_train, y_train[:, i])
        scores = cross_val_score(reg, x_test, y_test[:, i], cv=5, scoring='neg_mean_squared_error')
        score = -scores.mean()
        if score > high_score:
            high_score = score
            label = label_col[i]
        x_new = util.add_intercept(x_test)
        pred = x_new.dot(theta[i, :])
        den = pd.DataFrame({'Actual': y_test[:, i],
                            'Prediction': pred, })
        p = den.plot.kde()
        fig = p.get_figure()
        fig.savefig("density_plot/" + label_col[i] + '_density.png')
        print('Multiple Linear Regression MSE for', label_col[i], 'is',
              score, '+-', scores.std()*2)
    print('In profile', profile, 'the highest score for', label, 'is',high_score)

    # if only one input, we can plot it
    if n_input == 1:
        for i in range(n_label):
            save_path = "plots/" + \
                        input_col[0] + '_vs_' + label_col[i] + '.png'
            util.plot(x_test, y_test[:, i], theta[i, :],
                      input_col[0], label_col[0], save_path)
    print("theta is: ")
    print(theta)

def kNeighbor(X_train, Y_train, X_test, Y_test):
    from sklearn.neighbors import KNeighborsRegressor
    import math
    rmse_val = []  # to store rmse values for different k
    for K in range(20):
        K = K + 1
        model = KNeighborsRegressor(n_neighbors=K)

        model.fit(X_train, Y_train)  # fit the model
        pred = model.predict(X_test)  # make prediction on test set
        error = math.sqrt(mean_squared_error(Y_test, pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
        print('RMSE value for k= ', K, 'is:', error)

if __name__ == '__main__':
    input_col = ["ambient", "coolant", "motor_speed", "i_d", "i", "u"]
    label_col = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
    path = 'profile_data/Profile_4.csv'
    main(profile=4,
        input_col=input_col,
        label_col=label_col,
        ransac=False,
        cross = False,  # cross test: True
        profile_test=6) # if cross test, profile used for test
