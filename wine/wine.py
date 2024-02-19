import sys
sys.path.append("../")
from mlp import MLP
import numpy as np

with open("./wine.data", "r") as f:
	lines = f.readlines()  # list of strings

tokens = lines[0].split(',')
n = len(lines)
d = len(tokens) - 1

print("n=" + str(n))
print("d=" + str(d))

# allocating tensors for the dataset
X = np.zeros((n,d), dtype=np.float32)
y = np.zeros(n, dtype=np.int32)

# filling up tensors using the data on file
for i, line in enumerate(lines):
	tokens = line.split(',')
	y[i] = int(tokens[0]) - 1  # the first token is class label

	for j in range(0, d):
		X[i,j] = float(tokens[j+1])

# normalizing data
X = X - np.mean(X, axis=0)
X = X / np.std(X, axis=0)

# split X into train and test sets
indices = np.arange(n)
np.random.shuffle(indices)
X = X[indices,:]
#print(X.shape)
y = y[indices]
y = y.reshape(y.shape[0], 1)
#print(y.shape)

y_hot = np.zeros((y.shape[0],3))
#print(y_hot.shape)
for i in range(len(y)):
	y_hot[i][y[i][0]] = 1

'''	
for i in range(len(y)):
	print(y[i])
	print(y_hot[i])
'''
'''

X_train = X[0:int(n*0.75),:]
y_train = y[0:int(n*0.75)]
X_val = X[int(n*0.75):,:]
y_val = y[int(n*0.75):]

# creating neural network
net = MLP(13, [[32, "sigmoid"],[64, "sigmoid"], [64, "sigmoid"], [64, "sigmoid"],[1, "linear"]], loss="mse", log_rate=50)

net.train(X_train, y_train, batch_size=0, epochs=30000, lr=0.00005, X_Val=X_val, Y_Val=y_val, patience=100, plot=True, optimizer="adam")

'''


X_train = X[0:int(n*0.75),:]
y_train = y_hot[0:int(n*0.75):]
X_val = X[int(n*0.75):,:]
y_val = y_hot[int(n*0.75):,:]

net = MLP(13, [[32, "relu"],[64, "relu"], [64, "sigmoid"], [64, "relu"],[3, "softmax"]], loss="categorical_crossentropy", log_rate=50)



net.train(X_train, y_train, batch_size=0, epochs=30000, lr=0.00005, X_Val=X_val, Y_Val=y_val, patience=100, plot=True, optimizer="adam")

