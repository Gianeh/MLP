import sys
sys.path.append("../")
from mlp import MLP
import numpy as np

# Experiments on Wine dataset

# Loading data from file
with open("./wine.data", "r") as f:
	lines = f.readlines()  


# This iteration of the experiment uses the folowing combination of preprocessing and model parameters:


# Allocating tensors for the dataset
tokens = lines[0].split(',')
n = len(lines)
d = len(tokens) - 1
X = np.zeros((n,d), dtype=np.float32)
y = np.zeros(n, dtype=np.int32)

# Filling up tensors using the data from file
for i, line in enumerate(lines):
	tokens = line.split(',')
	y[i] = int(tokens[0]) - 1  # the first token is class label

	for j in range(0, d):
		X[i,j] = float(tokens[j+1])

# Data normalization
X = X - np.mean(X, axis=0)
X = X / np.std(X, axis=0)

# Split X into train and test sets
indices = np.arange(n)
np.random.shuffle(indices)
X = X[indices,:]
y = y[indices]
y = y.reshape(y.shape[0], 1)


# 1-hot encoding of targets
y_hot = np.zeros((y.shape[0],3))
for i in range(len(y)):
	y_hot[i][y[i][0]] = 1


# Training data
X_train = X[0:int(n*0.75),:]
y_train = y_hot[0:int(n*0.75):]

# Validation data
X_val = X[int(n*0.75):,:]
y_val = y_hot[int(n*0.75):,:]


# Model definition
net = MLP(13, [[32, "relu"],[64, "relu"], [64, "sigmoid"], [64, "relu"],[3, "softmax"]], loss="categorical_crossentropy", log_rate=50)


# Training
net.train(X_train, y_train, batch_size=0, epochs=30000, lr=0.00005, X_Val=X_val, Y_Val=y_val, early_stopping=True, patience=100, plot=True, optimizer="adam")
net.plot(save=True, name="wine_plot")
net.save_model("wine_model")

# Confusion matrix and accuracy
confusion = net.create_confusion_matrix(X=X_val, Y=y_val, name="wine_confusion_matrix")
net.print_confusion_matrix(confusion=confusion)
net.print_eval_metric(confusion=confusion, metric="accuracy")