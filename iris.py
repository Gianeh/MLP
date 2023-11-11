from neural_net import MLP
import numpy as np

# import the iris.data content into a numpy 5 dimensional array
with open("iris.data", "r") as f:
    # read line per line and remove the trailing newline character
    data = [line.rstrip() for line in f.readlines()]
    # split each line by the comma character
    data = [line.split(',') for line in data]
    # transform first 4 elements of each line into a float
    data = [[float(x) for x in line[:4]] + [line[4]] for line in data]

# convert the list of lists into a numpy array
data = np.array(data)


# one-hot encode the labels
classes = np.unique(data[:, 4])
one_hot = np.zeros((data.shape[0], len(classes)))

for i, label in enumerate(data[:, 4]):
    one_hot[i, np.where(classes == label)] = 1

# Remove the 5th column
data = np.delete(data, 4, axis=1)

# Add the one-hot encoded data
data = np.hstack((data, one_hot))

# convert the data to float
data = data.astype(np.float32)

# shuffle the data
np.random.shuffle(data)

print(data)

# split the data into training and test sets
train_data = data[:150, :4]
train_target = data[:130, 4:].reshape(-1, 1)

test_data = data[150:, :4]
test_target = data[150:, 4:].reshape(-1, 1)

# create the network
net = MLP(4, 3, [4], activation="sigmoid", loss="categorical_crossentropy", log_rate=50)

# print the network
net.print_net()

# train the network
net.train(train_data, train_target, batch_size=50, epochs=40000, lr=0.042)

# try to predict 6.2,3.4,5.4,2.3,Iris-virginica
O = net.predict(np.array([[6.2,3.4,5.4,2.3]]))  # note, the double square brackets are needed to create a 2D array and treat it as a batch of 1 element
print(O[-1])
print("predicted class: {}".format(np.argmax(O[-1])))
print("real class: {}".format(2))
'''
# test the network
O = net.evaluate(test_data, test_target)
print("test loss: {}".format(O))

'''
## Accuracy lacks because of the lack of an optimizer, either lr is too high or epochs are too low