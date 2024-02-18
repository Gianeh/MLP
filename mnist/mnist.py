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

# create the model
net = MLP(784, [[784, "sigmoid"],[784, "sigmoid"], [32, "sigmoid"], [10, "softmax"]], loss="categorical_crossentropy", log_rate=1)

# load the pre-trained model
#net.load_model("mnist_model")

# train the model
net.train(X=train_images, Y=train_labels, X_Val=test_images, Y_Val=test_labels, batch_size=30000, epochs=10, lr=0.00005, optimizer="adam", plot=True)

# save the model
#net.save_model("mnist_model") -


# perform some inference on a portion of the test data and show results
accuracy = 0
for test in range(len(test_images)):
    pred = net.predict(test_images[test])
    print("Predicted: ", pred.T)
    print("Actual: ", test_labels[test].T)
    if np.argmax(pred) == np.argmax(test_labels[test]):
        print("Correct!")
        accuracy += 1
    else:
        print("Incorrect!")

print("Accuracy: ", accuracy/len(test_images) * 100, "%")

#net.plot()