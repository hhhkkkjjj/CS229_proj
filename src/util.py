import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Add intercept to matrix x.
    Args:
        x: 2D NumPy array.
    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, input_col, label_col):
    """Load dataset from a CSV file.
    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.
    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """


    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i] in input_col]
    l_cols = [i for i in range(len(headers)) if headers[i] in label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)


    return inputs, labels

def plot(x, y, theta, x_name, y_name, save_path):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x, y, 'go', linewidth=0.5)

    # Plot fitting line
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    line_x = np.linspace(x_min, x_max, 10)
    line_y = theta[0] + theta[1] * line_x
    plt.plot(line_x, line_y, 'b-', linewidth=2)

    plt.xlim(x_min-.1, x_max+.1)
    plt.ylim(y_min-.1, y_max+.1)

    # Add labels and save to disk
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(save_path)