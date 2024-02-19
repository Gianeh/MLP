import numpy as np
import sys
sys.path.append("../")
from mlp import MLP


def load_data_from_file(file):
    data = np.loadtxt(file, dtype=np.float32)               # loading data from file (targets are on the last column)

    # Splitting inputs and targets
    X = data[:, 0:-1]
    y = data[:, -1].reshape((X.shape[0], 1))
    return X, y



def create_splits_unbalanced(X, y, train_val_fractions=(0.5, 0.25), randomize=True):
    # Randomizing data
    n = X.shape[0]
    if randomize:
        indices = np.arange(n)
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]

    # Collecting indices of the examples that will fall into the training, validation and test sets
    n = X.shape[0]
    a = round(n * train_val_fractions[0])
    b = a + round(n * train_val_fractions[1])

    train_set_indices = np.arange(0, a)
    val_set_indices = np.arange(a, b)
    test_set_indices = np.arange(b, n)

    # Splitting into training, validation and test sets
    X_train = X[train_set_indices, :]
    y_train = y[train_set_indices]
    X_val = X[val_set_indices, :]
    y_val = y[val_set_indices]
    X_test = X[test_set_indices, :]
    y_test = y[test_set_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test



def normalize_data(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X)
    if std is None:
        std = np.std(X)
    return (X - mean) / std, mean, std



if __name__ == "__main__":

    data_X, data_y = load_data_from_file('./housing.data')

    # Data splitting
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test = \
        create_splits_unbalanced(data_X, data_y, train_val_fractions=[0.7, 0.15], randomize=True)

    # Normalizing data
    data_X_train, m, s = normalize_data(data_X_train)
    data_X_val, _, _ = normalize_data(data_X_val, m, s)
    data_X_test, _, _ = normalize_data(data_X_test, m, s)

    net = MLP(13, [[32, "sigmoid"],[64, "sigmoid"], [64, "sigmoid"], [64, "sigmoid"],[1, "linear"]], loss="mse", log_rate=50)


    # Training phase

    net.train(data_X_train, data_y_train, batch_size=0, epochs=1000, lr=0.00005, X_Val=data_X_val, Y_Val=data_y_val, patience=100, plot=True, optimizer="adam")
    net.plot(save=True)
    net.save_model()

    
    # Evaluation phase

    net.load_model()
    predictions = net.predict(data_X_val)
    targets = data_y_val

    for i in range(targets.shape[0]):
        print('Target : ', targets[i][0], ' , Prediction : ' , predictions[0][i], '\n')

    
