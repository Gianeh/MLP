import deeplake
import sys
sys.path.append("../")
from mlp import MLP
import numpy as np
import os

if not os.path.exists("./mnist_train_data.txt"):
    train = deeplake.load("hub://activeloop/mnist-train")
    test = deeplake.load("hub://activeloop/mnist-test")

    # extract tensors from dataset
    train_images = train['images']
    train_labels = train['labels']

    test_images = test['images']
    test_labels = test['labels']

    # aggregate data into a numpy array
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # flatten the images
    train_images = train_images.reshape(train_images.shape[0], -1)
    print("Train data has shape: \n", train_images.shape)
    test_images = test_images.reshape(test_images.shape[0], -1)
    print("Test data has shape: \n", test_images.shape)
    tst_labels = train_labels.reshape(train_labels.shape[0], -1)

    trn_labels = test_labels.reshape(test_labels.shape[0], -1)

    train_labels = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels.shape[0]):
        train_labels[i][tst_labels[i]] = 1
    print("Train labels has shape: \n", train_labels.shape)

    test_labels = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels.shape[0]):
        test_labels[i][trn_labels[i]] = 1
    print("Test labels has shape: \n", test_labels.shape)

    # save the dataset files
    np.savetxt("./mnist_train_data.txt", train_images)
    np.savetxt("./mnist_train_labels.txt", train_labels)
    np.savetxt("./mnist_test_data.txt", test_images)
    np.savetxt("./mnist_test_labels.txt", test_labels)

else:
    train_images = np.loadtxt("./mnist_train_data.txt")
    train_labels = np.loadtxt("./mnist_train_labels.txt")
    test_images = np.loadtxt("./mnist_test_data.txt")
    test_labels = np.loadtxt("./mnist_test_labels.txt")

# Grid search
MAX_EPOCHS = 200
    
# 3 Hidden layers
archs = [[784, [800, "a"], [900, "a"], [128, "a"], [10, "softmax"]], [784, [850, "a"], [950, "a"], [156, "a"], [10, "softmax"]], [784 ,[900, "a"], [1000, "a"], [200, "a"], [10, "softmax"]]]
activations = ["sigmoid", "tanh"]
optimizers = ["adam"]
batch_sizes = [0, 10]

for arch in archs:
    for a1 in activations:
        for a2 in activations:
            for a3 in activations:
                architecture = [arch[0], arch[1], arch[2], arch[3]]
                architecture[1][1] = a1
                architecture[2][1] = a2
                architecture[3][1] = a3
                net = MLP(architecture, loss="categorical_crossentropy", log_rate=1)
                for opt in optimizers:
                    for bs in batch_sizes:
                        # SINGLE TRAINING
                        for i in range(MAX_EPOCHS):
                            train_indices = np.random.randint(0, train_images.shape[0], 6000)
                            train_data = train_images[train_indices]
                            train_targets = train_labels[train_indices]
                            test_indices = np.random.randint(0, test_images.shape[0], 1000)
                            test_data = test_images[test_indices]
                            test_targets = test_labels[test_indices]
                            net.train(train_data, train_targets, batch_size=bs, epochs=1, lr=0.0001, X_Val=test_data, Y_Val=test_targets, plot=False, optimizer=opt)
                        print("Architecture: ", architecture, " Activation functions: ", a1, a2, a3, " Optimizer: ", opt, " Batch size: ", bs, " Loss: ", net.evaluate(test_data, test_targets))

    
# 4 Hidden layers