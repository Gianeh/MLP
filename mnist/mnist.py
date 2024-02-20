import deeplake
import sys
sys.path.append("../")
from mlp import MLP
import numpy as np
import os

# Experiments on MNIST dataset

# If the dataset is not present, download it from the hub
if not os.path.exists("./mnist_train_data.txt"):
    train = deeplake.load("hub://activeloop/mnist-train")
    test = deeplake.load("hub://activeloop/mnist-test")

    # Extract tensors from dataset
    train_images = train['images']
    train_labels = train['labels']

    test_images = test['images']
    test_labels = test['labels']

    # Aggregate data into a numpy array
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Flatten the images
    train_images = train_images.reshape(train_images.shape[0], -1)
    print("Train data has shape: \n", train_images.shape)
    test_images = test_images.reshape(test_images.shape[0], -1)
    print("Test data has shape: \n", test_images.shape)

    tst_labels = train_labels.reshape(train_labels.shape[0], -1)

    trn_labels = test_labels.reshape(test_labels.shape[0], -1)

    # 1-hot encoding of targets
    train_labels = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels.shape[0]):
        train_labels[i][tst_labels[i]] = 1
    print("Train labels has shape: \n", train_labels.shape)

    test_labels = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels.shape[0]):
        test_labels[i][trn_labels[i]] = 1
    print("Test labels has shape: \n", test_labels.shape)

    # Save the dataset files
    np.savetxt("./mnist_train_data.txt", train_images)
    np.savetxt("./mnist_train_labels.txt", train_labels)
    np.savetxt("./mnist_test_data.txt", test_images)
    np.savetxt("./mnist_test_labels.txt", test_labels)

# If the dataset is present, load it
else:
    train_images = np.loadtxt("./mnist_train_data.txt")
    train_labels = np.loadtxt("./mnist_train_labels.txt")
    test_images = np.loadtxt("./mnist_test_data.txt")
    test_labels = np.loadtxt("./mnist_test_labels.txt")

# Create the model
net = MLP(784, [[784, "tanh"],[784, "sigmoid"], [32, "sigmoid"], [10, "softmax"]], loss="categorical_crossentropy", log_rate=1)

# Load the pre-trained model
# net.load_model("mnist_model")

# Train the model
net.train(X=train_images, Y=train_labels, X_Val=test_images, Y_Val=test_labels, batch_size=0, epochs=100, lr=0.00005, optimizer="adam")
net.plot()
# Save the model
# net.save_model("mnist_model") -

# Printing Confusion matrix and accuracy
# net.print_confusion_matrix(test_images, test_labels, name="mnist_evaluation")
# net.print_eval_metric(test_images, test_labels, metric="accuracy")