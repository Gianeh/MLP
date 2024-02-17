import deeplake
from mlp import MLP
import numpy as np


train = deeplake.load("hub://activeloop/mnist-train")
test = deeplake.load("hub://activeloop/mnist-test")

# extract tensors from dataset
train_images = train['images']
train_labels = train['labels']

# aggregate data into a numpy array
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# flatten the images
train_images = train_images.reshape(train_images.shape[0], -1)
labels = train_labels.reshape(train_labels.shape[0], -1)
train_labels = np.zeros((train_labels.shape[0], 10))
for i in range(train_labels.shape[0]):
    train_labels[i][labels[i]] = 1
print(train_labels[0])
print(labels[0])

net = MLP(784, [[512, "sigmoid"],[1024, "sigmoid"], [64, "sigmoid"], [10, "softmax"]], loss="categorical_crossentropy", log_rate=50)
#net.train(train_images, train_labels, batch_size=0, epochs=100, lr=0.00005, optimizer="adam", plot=True)
net.plot()