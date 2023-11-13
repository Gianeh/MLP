# another boolean problem but in n dimensions
from neural_net import MLP
import numpy as np
import matplotlib.pyplot as plt

# create a circular dataset

# Number of points in each ring
n_points = 5000

# For the ring with radius 1
theta = np.random.uniform(0, 2*np.pi, n_points)
r = np.random.uniform(0.6, 1.1, n_points)
x1 = 0.5 + r * np.cos(theta)
y1 = 0.5 + r * np.sin(theta)
labels1 = np.zeros(n_points)  # Label 0 for outer ring

# For the ring with radius 0.3
theta = np.random.uniform(0, 2*np.pi, n_points)
r = np.random.uniform(-0.1, 0.45, n_points)
x2 = 0.5 + r * np.cos(theta)
y2 = 0.5 + r * np.sin(theta)
labels2 = np.ones(n_points)  # Label 1 for inner ring

# Combine the points from both rings
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
labels = np.concatenate((labels1, labels2))

# Combine x, y coordinates and labels into a 2D array
data = np.column_stack((x, y, labels))
np.random.shuffle(data)

# train a neural network to guess what point is in the outer or in the inner ring
net = MLP(2,1,[3,3], log_rate=5, loss="mae")

# split data into input/label
training_data = data[:, :2]
training_target = data[:, 2:].reshape(-1,1)

# train the network
net.train(training_data, training_target, batch_size=100, epochs=3000, lr=0.03)

# plot a series of prediction to plot a map of neural function
plt.figure(figsize=(6, 6))

map = []
for i in range(50):
    for j in range(50):
        map.append([i/25-0.5, j/25-0.5, net.predict(np.array([i/25-0.5, j/25-0.5]))[-1]])
map = np.array(map)

# plot map
# Create a color map
colors = ['g' if label > 0.5 else 'r' for label in map[:, 2]]

# plot map
plt.scatter(map[:, 0], map[:, 1], c=colors)

# plot data
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
plt.show()
plt.savefig("circle.png")