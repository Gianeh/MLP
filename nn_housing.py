import numpy as np
from mlp import MLP


def load_data_from_file(file):
    # loading data from file (regression targets are on the last column)
    Xy = np.loadtxt(file, dtype=np.float32)

    # splitting data and targets
    X = Xy[:, 0:-1]
    y = Xy[:, -1].reshape((X.shape[0], 1))
    return X, y


def create_splits_unbalanced(X, y, train_val_fractions=(0.5, 0.25), randomize=True):
    # randomizing data
    n = X.shape[0]
    if randomize:
        indices = np.arange(n)
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]

    # collecting indices of the examples that will fall into the training, validation and test sets
    n = X.shape[0]
    a = round(n * train_val_fractions[0])
    b = a + round(n * train_val_fractions[1])

    train_set_indices = np.arange(0, a)
    val_set_indices = np.arange(a, b)
    test_set_indices = np.arange(b, n)

    # splitting into training, validation and test sets
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



# entry point
if __name__ == "__main__":

    data_X, data_y = load_data_from_file('./housing/housing.data')

    # splitting
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test = \
        create_splits_unbalanced(data_X, data_y, train_val_fractions=[0.7, 0.15])

    # normalizing data
    data_X_train, m, s = normalize_data(data_X_train)
    data_X_val, _, _ = normalize_data(data_X_val, m, s)
    data_X_test, _, _ = normalize_data(data_X_test, m, s)
    #print(data_X_train.shape)

    net = MLP(13, [[32, "leaky_relu"],[64, "relu"],[1, "leaky_relu"]], loss="mse", log_rate=10)
    
    #Training phase

    #net.train(data_X_train, data_y_train, batch_size=10, epochs=10000, lr=0.00005)
    
    #net.save_model()
    '''
    #Evaluation phase
    net.load_model()
    #print(net.evaluate(data_X_val, data_y_val))
    predictions = net.predict(data_X_val)
    targets = data_y_val

    for i in range(targets.shape[0]):
        print('Target : ', targets[i][0], ' , Prediction : ' , predictions[0][i], '\n')
    '''
    
