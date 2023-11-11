from neural_net import MLP
import numpy as np
from matplotlib import pyplot as plt

# create a dataset of the boolean function: (A XOR B) with some noise
data = []
# pick 10000 random x1,x2 values around zero and one
for i in range(10000):
    x1 = np.random.randint(0, 2)
    x2 = np.random.randint(0, 2)
    # add some noise to the data
    x1 = np.random.normal(x1, 0.05)
    x2 = np.random.normal(x2, 0.05)
    # add the data to the dataset
    data.append([x1, x2, int(round(x1) ^ round(x2))])
data = np.array(data)

# print data
'''
for l in data:
    print(l[0], l[1], l[2])
'''

# split the data into training and test sets
# first 800 examples, just element 0 and 1 (the input)
train_data = data[:8000, :2]
# first 800 examples, just element 2 (the target)
train_target = data[:8000, 2].reshape(-1, 1)
# last 200 examples, just element 0 and 1 (the input)
test_data = data[8000:, :2]
# last 200 examples, just element 2 (the target)
test_target = data[8000:, 2]

# print data X
print(train_data.shape)
print(train_data)

# create the network
net = MLP(2, 1, [2,4], activation="sigmoid", loss="mse", log_rate=50)

# print the network
net.print_net()

# train the network
net.train(train_data, train_target, batch_size=500, epochs=20000, lr=0.1)

# test the network
O = net.predict(np.array([1,1]))
print(O[-1])
# print("test loss: {}".format(net.loss(O[-1], test_target)))

# predict output for a uniform distribution of points in [-0.5,1.5]x[-0.5,1.5] and add them to the plot
map = []
for i in range(50):
    for j in range(50):
        map.append([i/25-0.5, j/25-0.5, net.predict(np.array([i/25-0.5, j/25-0.5]))[-1]])
map = np.array(map)

# plot map points
for l in map:
    plt.plot(l[0], l[1], 'go' if l[2] < 0.5 else 'yo')

# plot the data
for l in data:
    plt.plot(l[0], l[1], 'ro' if l[2] == 0 else 'bo')
        
plt.savefig('plot.png')    # save the fig after showing a plot of predicted values