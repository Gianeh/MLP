
# Class MLP is designed to process row vector input and target, in general MLP takes as input a matrix where each row is one train sample
# and as target a matrix where each row is one target vector

# Internally every calculation is treated according to the conventional algebraic rules (reshaping of the input arguments passed to the net)
# where each input is a column vector (forming a matrix for multiple samples) that produce an output that is a column vector too

import os
import sys
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle



class MLP:
    def __init__(self, num_input, layers =[], loss = "mse", log_rate = 100, gradient_clipping = sys.maxsize):

        self.num_input = num_input                                                          # components number of the input vector
        self.layers = layers                                                                # layers list (including the output layer) with associated activations
        self.net = []
        self.outputs = []                                                                   # outputs list (including the intemediate outputs of the hidden layers)
        self.activations = []                                                               # activation scores list (including the intemediate activation scores of the hidden layers)
        self.init_net()                                                                     # network initialization

        # Class variables used with RPROP and ADAM optimizers
        self.prev_G = None                  # previous gradient tensor (referring to the previous time step)
        self.m = None                       # initialization of the first order moment
        self.v = None                       # initialization of the second order moment

        self.loss = self.init_loss(loss)                                                    # loss function initialization
        self.activation_functions = self.init_activation_functions(layers)                  # activation functions initialization
        self.log_rate = log_rate                                                            # log frequency in number of epochs
        self.loss_name = loss                                                               # variable for log purpose

        self.gradient_clipping = gradient_clipping                                          # maximum value of each component of the gradient

        self.lr = 0                                                                         # learning rate initialization (used in update method)

        self.epoch_loss = []                                                                # list of loss values ​​along the epochs on the training set
        self.epoch_val_loss = []                                                            # list of loss values ​​along the epochs on the validation set



# INITIALIZERS


# Init_net initialize according to the uniform distribution all the matrices weights and all the bias vectors according the dimensions of the layers involved
# furthermore at the end of the loop over all the layers, it initializes the lists of outputs and activations of the net, for each layer as None value

    def init_net(self):
        # loop over layers
        for i in range(len(self.layers)):
            # input layer
            if i == 0:
                self.net.append((
                    np.random.uniform(-1,1, size=(self.layers[i][0], self.num_input)),
                    np.random.uniform(-1,1, size=(self.layers[i][0],1))
                ))
            # hidden layer or output layer
            elif i < len(self.layers):      
                self.net.append((
                    np.random.uniform(-1,1, size=(self.layers[i][0], self.layers[i-1][0])),
                    np.random.uniform(-1,1, size=(self.layers[i][0],1))
                ))
    
        self.outputs = [None] * (len(self.layers))                                          # outputs list initialized as None
        self.activations = [None] * (len(self.layers))                                      # activations list initialized as None



    # Loss function initialization
    def init_loss(self, loss):
        if loss == "mse":                                                                   # mean squared error 
            return self.mse
        elif loss == "mae":                                                                 # mean absolute error
            return self.mae
        elif loss == "categorical_crossentropy":                                            # categorical crossentropy
            return self.categorical_crossentropy
        elif loss == "binary_crossentropy":                                                 # binary crossentropy
            return self.binary_crossentropy
        else:
            raise ValueError("Invalid loss function")



    # Activation functions initialization    
    def init_activation_functions(self, layers):
        activation_functions = []
        for layer in layers:
            if layer[1] == "sigmoid":                                                       # sigmoid
                activation_functions.append(self.sigmoid)
            elif layer[1] == "relu":                                                        # rectified linear unit
                activation_functions.append(self.relu)
            elif layer[1] == "leaky_relu":                                                  # leaky rectified linear unit
                activation_functions.append(self.leaky_relu)
            elif layer[1] == "tanh":                                                        # hyperbolic tangent
                activation_functions.append(self.tanh)
            elif layer[1] == "softmax":                                                     # softmax
                activation_functions.append(self.softmax)
            elif layer[1] == "linear":                                                      # linear
                activation_functions.append(self.linear)
            else:
                raise ValueError("Invalid activation function")
        return activation_functions



    # Method to save the model parameters
    def save_model(self, model_name = "model"):
        with open(model_name+".pkl", "wb") as file:
            pickle.dump(self.net, file)
        print(f"Model saved as {model_name}.pkl")



    # Method to load the model parameters
    def load_model(self, model_name = "model"):
        with open(model_name+".pkl", "rb") as file:
            self.net = pickle.load(file)
        print(f"Model loaded from {model_name}.npz")



# ACTIVATION FUNCTIONS


# The sigmoid activation function is an element-wise function that takes as input the activation column vector (for a batch is a matrix), where:
# if grad=False return a column vector too that has for each component the sigmoidal computation of the corresponding argument component
# if grad=True return a column vector where the component i is sigmoid(A[i]) * (1.0-sigmoid(A[i]))

    def sigmoid(self, A, grad=False):
        if not grad:
            return 1.0 / (1.0 + np.exp(-A))
        else:
            # gradient computation
            return self.sigmoid(A) * (1.0 - self.sigmoid(A))



# Relu activation function initialize with all zeros a tensor called output that has the same dimensions as the activation tensor argument (A)
# if grad=False return a tensor where each component is computed by the relu scalar function (max(0.0, x))
# if grad=True return a tensor where each component is 0 or 1 (derivative of relu is 0 for negative activation value and 1 for null or positive value)

    def relu(self, A, grad=False):
        output = np.zeros(A.shape)
        if not grad:
            for col in range(0, output.shape[1]):                   
                for row in range(0, output.shape[0]):
                    output[row][col] = max(0.0, A[row][col])
            return output
        else:
            # gradient computation
            for col in range(0, output.shape[1]):
                for row in range(0, output.shape[0]):
                    output[row][col] = 1 if (A[row][col] > 0) else 0
            return output


# Leaky Relu activation function initialize with all zeros a tensor called output that has the same dimensions as the activation tensor argument (A)
# if grad=False return a tensor where each component is computed by the leaky relu scalar function (y=f(x) where y=x if x>=0 and y=mx if x<0 and m is negative_slope variable)
# if grad=True return a tensor where each component is neg_slope or 1 (derivative of leaky relu is neg_slope for negative activation value and 1 for null or positive value)
    
    def leaky_relu(self, A, neg_slope=0.01 ,grad=False):
        output = np.zeros(A.shape)
        if not grad:
            for col in range(0, output.shape[1]):
                for row in range(0, output.shape[0]):
                    output[row][col] = A[row][col] if (A[row][col] >= 0) else (A[row][col] * neg_slope)
            return output
        else:
            # gradient computation
            for col in range(0, output.shape[1]):
                for row in range(0, output.shape[0]):
                    output[row][col] = 1 if (A[row][col] >= 0) else neg_slope
            return output



# The hyperbolic tangent activation function is an element-wise function that takes as input the activation column vector (for batch is a matrix), where:
# if grad=False return a column vector too that has for each component the tanh computation of the corresponding argument component
# if grad=True return a column vector where each component is ->  1 - (tanh(a)^2)
   
    def tanh(self, A, grad=False):
        if not grad:
            return np.tanh(A)                                                       # numpy implementation of tanh trims the value to prevent overflow
        else:
            # gradient computation
            return (1.0 - self.tanh(A)**2)



# The softmax activation function is an activation where each output component is function of all the input arguments
# it is assumed that this activation is used only in output layer in combination with categorical categorical crossentropy loss
# given this assumption, it is not necessary to calculate the derivative of the activation function since it is integrated into the
# derivative of the loss function (not included in the loss function implementation, but taken into account in the backward and train method)
   
    def softmax(self, A):
        output = np.zeros(A.shape)
        for col in range(0, output.shape[1]):
            for row in range(0, output.shape[0]):
                output[row][col] = np.exp(A[row][col])
            normalizer = np.sum(output[: , col], axis=0)
            output[:, col] = (output[: , col] / normalizer)
        return output



# Linear activation function if grad=False return a argument activation tensor (A)
# if grad=True return a tensor where each component is 1 
    
    def linear(self, A, grad=False):
        if not grad:
            return A
        else:
            # gradient computation
            return np.ones(A.shape)



# LOSS FUNCTIONS
        

# The mse loss function calculates the error tensor (matrix if mini/full-batch and column vector if online), where:
# if grad=False return always a scalar value (vector with one element) that is the mean over all the dimension of the squared error tensor
# where err * err is an element wise multiplication so has the same dimensions of err column vector
# if grad=True return always a tensor that has the same shape as OL(predictions tensor) and Y(targets tensor) and where each component
# is normalized by mean (1/Y.shape[0] where Y.shape[0] is the dimensionality of each target)

    def mse(self, OL, Y, grad=False):
        err = OL - Y
        if not grad:
            return np.mean(err * err)
        else:
            # gradient computation
            return (2.0 / Y.shape[0]) * err



# The mae loss function calculates the absolute error (scalar) that is the mean over all the dimensions of the absolute value error tensor
# if grad = True the method returns a tensor with the same shape of err, where each component is -1 or 1 and is normalized
# by the scalar n (number of elements in the batch == Y.shape[0])

    def mae(self, OL, Y, grad=False):
        err = OL - Y
        if not grad:
            return np.mean(np.abs(err))
        else:
            # gradient computation
            return np.sign(err)/Y.shape[0]
        


# Binary cross-entropy loss function is used only if the target is a scalar belonging to (0,1) (where the assumpion is that the output layer must have only one neuron)
# if grad = True the method returns a vector (batch mode) or a scalar (online mode) that has the same shape of OL (prediction) and Y (target)
# all the mathematical operations are element-wise (1-Y as the shape of Y due to the broadcasting)
        
    def binary_crossentropy(self, OL, Y, grad=False):
        if not grad:
            epsilon = 1e-15                                         # Small scalar to avoid numerical instability for log(0)
            OL = np.clip(OL, epsilon, 1 - epsilon)                  # Clip predicted values to avoid log(0)
            return -np.mean(Y * np.log10(OL) + (1 - Y) * np.log10(1 - OL))
        else:
            # gradient computation
            epsilon = 1e-15  
            OL = np.clip(OL, epsilon, 1 - epsilon)  
            return (OL - Y) / (OL * (1 - OL))



# The categorical crossentropy loss assume 1-hot encoded target for classification task.
# After limiting the output (OL) in a continuous set (0,1) with extrema excluded, the definition of crossentropy 
# is applied and it returns a row vector where each element refers to a training sample.
# The average of the losses over all the samples is therefore returned
# Remark: It is not necessary to define the derivative of the loss since, for the same reasons as in the softmax activation, it is assumed that softmax and categorical crossentropy
# are used exclusively together, and therefore the derivative of the loss is taken into account in the calculation of the output layer delta in the backward method (and train method too)
    
    def categorical_crossentropy(self, OL, y):
        epsilon = 1e-15                                             # Small constant to avoid numerical instability
        clipped_OL = np.clip(OL, epsilon, 1 - epsilon)              # Clip values to avoid log(0) or log(1)
        loss = -np.sum(y * np.log10(clipped_OL), axis=0)  
        loss = np.mean(loss)
        return loss



# NETWORK CORE ROUTINES - FORWARD, BACKWARD, UPDATE, TRAIN, PREDICT, EVALUATE


    # Forward method for each layer compute the activation scores and the output tensors and collects them in two separated lists
    def forward(self, X):
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            # input layer
            if l == 0:
                self.activations[l] = np.matmul(Wl, X) + bl
            # hidden layer or output layer
            else:
                self.activations[l] = np.matmul(Wl, self.outputs[l - 1]) + bl
            # activation function computation
            self.outputs[l] = self.activation_functions[l](self.activations[l])
        return self.outputs, self.activations



# Backward method for each layer compute the corresponding delta tensor (matrix for full/mini-batch and column vector for online) and uses it
# for the calculation of the gradient tensor (matrix for weights and column vector for biases)
# the product in Delta calculation is element wise (implies that both argument tensors have the same shape)
# at the end of each loop iteration (through layers) matrix (weights) and vector (biases) gradients are collected in a list of tuples G[]
# Remark: extra argument Y wad added to simplify the implementation of categorical crossentropy so softmax function does not have to explicitly calculate the gradients

    def backward(self, X, O, A, loss_derivative, Y):
        L = len(self.net)
        G = [None] * L
        for l in range(L-1, -1, -1):
            # output layer
            if l == L-1:
                # is softmax is used in output layer
                if self.layers[l][1] == "softmax":
                    Delta = O[-1] - Y
                else:
                    Delta = self.activation_functions[l](A[l], grad=True) * loss_derivative
            # hidden layers
            else:
                Wl_plus_1, _ = self.net[l + 1]
                Delta = self.activation_functions[l](A[l], grad=True) * (np.matmul(Wl_plus_1.T, Delta))
            if l > 0:
                Wl_grad = np.matmul(Delta, O[l - 1].T)
            else:
                Wl_grad = np.matmul(Delta, X.T)
            bl_grad = np.sum(Delta, axis=1).reshape(self.net[l][1].shape[0], 1)         # sum reduction over the samples dimension (axis=1) 
            G[l] = (Wl_grad, bl_grad)                                                   # gradient tensor for layer l
        return G



    # Update method which defines the network parameters update rule also including learning rate optimizers
    def update(self, G, lr, optimizer):
        # Basic Adaptive Learning Rate
        if optimizer == "basic":
            lr_interval = [0.01*lr, 10*lr]

            if len(self.epoch_loss) < 2:                                                # skip the first update
                pass
            elif self.epoch_loss[-1] > self.epoch_loss[-2]:
                self.lr = np.clip(self.lr*0.75, lr_interval[0], lr_interval[1])
            else:
                self.lr = np.clip(self.lr*1.05, lr_interval[0], lr_interval[1])
        
        # Resilient Backpropagation - RPROP
        elif optimizer == "rprop":
            if self.prev_G is not None:
                for l in range(len(self.net)):
                    Wl, bl = self.net[l]
                    Wl_grad, bl_grad = G[l]
                    # clip gradients
                    Wl_grad = np.clip(Wl_grad, -self.gradient_clipping, self.gradient_clipping)
                    bl_grad = np.clip(bl_grad, -self.gradient_clipping, self.gradient_clipping)

                    Wl_prev_grad, bl_prev_grad = self.prev_G[l]
                    Wl_sign = np.sign(Wl_grad * Wl_prev_grad)
                    bl_sign = np.sign(bl_grad * bl_prev_grad)
                    Wl = Wl + (Wl_sign * lr * np.sign(-Wl_grad))
                    bl = bl + (bl_sign * lr * np.sign(-bl_grad))
                    self.net[l] = (Wl, bl)
            self.prev_G = G
            return
        
        # Adaptive Moment Estimation - ADAM
        elif optimizer == "adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            if self.m is None:
                self.m = []
                self.v = []
                for l in self.net:
                    w, b = l
                    self.m.append([np.zeros(w.shape), np.zeros(b.shape)])
                    self.v.append([np.zeros(w.shape), np.zeros(b.shape)])

            for l in range(len(self.net)):
                Wl, bl = self.net[l]
                Wl_grad, bl_grad = G[l]
                
                # clip gradients
                Wl_grad = np.clip(Wl_grad, -self.gradient_clipping, self.gradient_clipping)
                bl_grad = np.clip(bl_grad, -self.gradient_clipping, self.gradient_clipping)
                
                self.m[l][0] = beta1 * self.m[l][0] + (1 - beta1) * Wl_grad
                self.m[l][1] = beta1 * self.m[l][1] + (1 - beta1) * bl_grad
                self.v[l][0] = beta2 * self.v[l][0] + (1 - beta2) * Wl_grad**2
                self.v[l][1] = beta2 * self.v[l][1] + (1 - beta2) * bl_grad**2
                
                # Bias-corrected estimates of first and second order moments 
                m_hat_w = self.m[l][0] / (1 - beta1)
                m_hat_b = self.m[l][1] / (1 - beta1)
                v_hat_w = self.v[l][0] / (1 - beta2)
                v_hat_b = self.v[l][1] / (1 - beta2)

                Wl -= lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
                bl -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
                
                self.net[l] = (Wl, bl)
            return

        # Standard Update Rule
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            Wl_grad, bl_grad = G[l]
            # clip gradients
            Wl_grad = np.clip(Wl_grad, -self.gradient_clipping, self.gradient_clipping)
            bl_grad = np.clip(bl_grad, -self.gradient_clipping, self.gradient_clipping)
            Wl -= self.lr * Wl_grad
            bl -= self.lr * bl_grad
            self.net[l] = (Wl, bl)



    # In train method X and Y inputs are transposed in order to process row-element inputs and targets as in the algebraic rule fashion
    def train(self, X, Y, batch_size=0, epochs=100, lr=0.001, X_Val=None, Y_Val=None, plot=False, early_stopping=False, patience=1, optimizer=None):
        self.lr = lr
        # Adjust maximum gradient_clipping value in case of high learning rate
        if self.lr > 1: self.gradient_clipping /= self.lr

        X=X.T                                                           # input transposition 
        Y=Y.T                                                           # target transposition
        if plot:
            plt.ion()
            fig, ax = plt.subplots()
        # if batch size is 0, train on the whole dataset
        if batch_size == 0:
            batch_size = len(X.T)

        # Training loop
        for epoch in range(epochs):
            avg_loss = 0                                                # average loss of the current epoch

            if X_Val is not None and Y_Val is not None:                 # if validation examples have been provided
                val_loss = self.evaluate(X_Val, Y_Val)                  # loss calculation on validation data
                if epoch % self.log_rate == 0:
                    print(f"Epoch {epoch} - Validation Loss: {val_loss}")
                self.epoch_val_loss.append(val_loss)                    # collect the validation loss

            # Batch loop
            for i in range(0, len(X.T), batch_size):
                O, A = self.forward(X[:,i:i+batch_size])
                if self.layers[-1][1] == "softmax":
                    G = self.backward(X[:,i:i+batch_size], O, A, loss_derivative = 0, Y = Y[:,i:i+batch_size])
                else:
                    if self.loss == self.categorical_crossentropy:
                        print("Categorical crossentropy loss function requires softmax activation in the output layer")
                        exit()
                    loss_derivative = self.loss(O[-1], Y[:,i:i+batch_size], grad = True)
                    G = self.backward(X[:,i:i+batch_size], O, A, loss_derivative = loss_derivative, Y = 0)
                if epoch % self.log_rate == 0:
                    print(f"Epoch {epoch} - Batch {int(i/batch_size)} - Loss: {self.loss(O[-1], Y[:,i:i+batch_size])}")

                avg_loss += self.loss(O[-1], Y[:,i:i+batch_size])       # adding partial loss (over the batch)

                self.update(G, lr=lr, optimizer=optimizer)              # update step

            self.epoch_loss.append(avg_loss/(len(X.T)/(batch_size+1)))  # collect the training loss

            if early_stopping and X_Val is not None and Y_Val is not None:
                if self.early_stopping(patience):                       # early stopping criterion
                    break
            
            # Plot current epoch's loss
            if epoch % self.log_rate == 0 and plot: self.plot(fig = fig, ax = ax)

        # Close plot
        if plot:
            plt.close(fig)
            plt.ioff()



    # Predict method that takes as input a row vector (or a matrix with row elements) and returns the output of the network
    def predict(self, X):
        if(X.ndim == 1):
            X = X.reshape(X.shape[0],1)         # reshaping as a column vector (when X is a row vector)
        else:
            X=X.T                               # transposing the input matrix (when X is a matrix with row elements)
        O, _ = self.forward(X)
        return O[-1]



    # Evaluate method that takes as input a matrix of input and a matrix of targets and returns the loss value of the predictions
    def evaluate(self, X, Y):
        if(X.ndim == 1):
            X=X.reshape(X.shape[0],1)           # reshaping as a column vector (when X is a row vector)
            Y=Y.reshape(Y.shape[0],1)           # reshaping as a column vector (when Y is a row vector)
        else:
            X=X.T                               # transposing the input matrix (when X is a matrix with row elements)
            Y=Y.T                               # transposing the input matrix (when Y is a matrix with row elements)
        O, _ = self.forward(X)
        return self.loss(O[-1], Y)              # loss computation
    


    # Early stopping method based on the network loss function
    def early_stopping(self, patience):
        if len(self.epoch_val_loss) > patience:
            # if for patience epochs the loss is increasing stop the training
            if np.all(np.array(self.epoch_val_loss[-patience+1:]) > np.array(self.epoch_val_loss[-patience])):
                print(f"Early stopping at epoch {len(self.epoch_val_loss)}")
                return True
        return False



    # EVALUATION METRICS


    # Method to build the confusion matrix for each class
    def create_confusion_matrix(self,X,Y):
        classes = [ None ] * Y.shape[1]                         # list of confusion matrices initialized to None
        for cls in range(len(classes)):                         # loop over classes
            tp=0                                                # true positives counter
            tn=0                                                # true negatives counter
            fp=0                                                # false positives counter
            fn=0                                                # false negatives counter
            for target_index in range(len(Y.T[0])):             # loop over targets
                prediction = self.predict(X=X[target_index])
                if np.argmax(Y[target_index]) == cls and np.argmax(prediction) == cls:
                    tp += 1
                elif np.argmax(Y[target_index]) != cls and np.argmax(prediction) == cls:
                    fp += 1
                elif np.argmax(Y[target_index]) == cls and np.argmax(prediction) != cls:
                    fn += 1
                elif np.argmax(Y[target_index]) != cls and np.argmax(prediction) != cls:
                    tn += 1
                else:
                    print("Error")

            classes[cls] = [tp,tn,fp,fn]

        return classes
    


    # Method to print the confusion matrix
    def print_confusion_matrix(self,X=None,Y=None, name=None, confusion=None):
        if confusion is None:                                                           # confusion matrix not provided as input
            if X is not None and Y is not None:
                data = self.create_confusion_matrix(X, Y)                               # construction of the confusion matrix
            else : print('Not enough data to compute metric')
        else : data = confusion                                                         # confusion matrix was given as input

        # Fancy ascii confusion matrix
        for cls in range(len(data)):                                                    # loop over classes
            print(f"Confusion Matrix for class {cls}:")
            print("\n\t\t\tPredicted Class\n")
            print("\t\t==================================")
            # ||           ||          ||
            print("\t\t||", end="")
            print("\t\t||\t\t||")
            # ||    TP     ||    FN    ||
            print("\t\t||\tTP\t||\tFN\t||")
            # ||    tp     ||    fn    ||
            print("\t\t||", end="")
            print(f"\t{data[cls][0]}\t||\t{data[cls][3]}\t||")
            # ||           ||          ||
            print("\t\t||\t\t||\t\t||")
            print("True Class", end="")
            print("\t==================================")
            # ||           ||          ||
            print("\t\t||", end="")
            print("\t\t||\t\t||")
            # ||    FP     ||    TN    ||
            print("\t\t||\tFP\t||\tTN\t||")
            # ||    fp     ||    tn    ||
            print("\t\t||", end="")
            print(f"\t{data[cls][2]}\t||\t{data[cls][1]}\t||")
            # ||           ||          ||
            print("\t\t||\t\t||\t\t||")
            print("\t\t==================================")
            print("\n\n")

            # dump same strings to file
            if name is not None:
                with open(name+'.txt', "a") as file:
                    file.write(f"\t\t\t\tConfusion Matrix for class {cls}:\n")
                    file.write("\n \t\t\t\tPredicted Class\n\n")
                    file.write("\t\t\t==================================\n")
                    file.write("\t\t\t||\t\t\t\t||\t\t\t\t||\n")
                    file.write("\t\t\t||\t\tTP\t\t||\t\tFN\t\t||\n")
                    file.write(f"\t\t\t||\t\t{data[cls][0]}\t\t||\t\t{data[cls][3]}\t\t||\n")
                    file.write("\t\t\t||\t\t\t\t||\t\t\t\t||\n")
                    file.write("True Class\t==================================\n")
                    file.write("\t\t\t||\t\t\t\t||\t\t\t\t||\n")
                    file.write("\t\t\t||\t\tFP\t\t||\t\tTN\t\t||\n")
                    file.write(f"\t\t\t||\t\t{data[cls][2]}\t\t||\t\t{data[cls][1]}\t\t||\n")
                    file.write("\t\t\t||\t\t\t\t||\t\t\t\t||\n")
                    file.write("\t\t\t==================================\n")
                    file.write("\n\n")
                    


    # Method that outputs a specific evaluation metrics given the argument metric (accuracy, precision, recall, f1) or all the metrics if is None
    def get_eval_metric(self, X=None, Y=None, metric=None, confusion=None):
        if confusion is None:                                                            # confusion matrix not provided as input
            if X is not None and Y is not None:
                data = self.create_confusion_matrix(X, Y)                                # construction of the confusion matrix
            else : print('Not enough data to compute metric')
        else : data = confusion                                                          # confusion matrix was given as input

        metrics = [] 
        if metric == 'accuracy':                                                         # accuracy metric
            for cls in data:
                metrics.append((cls[0] + cls[1]) / (cls[0] + cls[1] + cls[2] + cls[3]))  
        elif metric == 'precision':                                                      # precision metric
            for cls in data:
                metrics.append((cls[0]) / (cls[0] + cls[2]))
        elif metric == 'recall':                                                         # recall metric
            for cls in data:
                metrics.append((cls[0]) / (cls[0] + cls[3]))
        elif metric == 'f1':                                                             # f1 metric
            precision = np.array(self.get_eval_metric(metric = 'precision', confusion = data))
            recall = np.array(self.get_eval_metric(metric = 'recall', confusion = data))
            metrics = list((2 * precision * recall) / (precision + recall))
        else:                                                                            # no argument metric provided -> all metrics
            metrics.append(self.get_eval_metric(metric = 'accuracy', confusion = data))
            metrics.append(self.get_eval_metric(metric = 'precision', confusion = data))
            metrics.append(self.get_eval_metric(metric = 'recall', confusion = data))
            metrics.append(self.get_eval_metric(metric = 'f1', confusion = data))

        return metrics
        


    # Method to print the elements of the confusion matrix for all classes
    def print_eval_metric(self, X=None, Y=None, confusion=None):
        metrics = self.get_eval_metric(X=X, Y=Y, confusion=confusion)                   # confusion matrix computation
        for i in range(len(metrics[0])):                                                # loop over classes
            print('Class ', i, ':')
            print('Accuracy = \t', metrics[0][i])
            print('Precision = \t', metrics[1][i])
            print('Recall = \t', metrics[2][i])
            print('F1 = \t\t', metrics[3][i])
            print()
        


    # Method to plot training and validation losse over epochs
    def plot(self, save=False, fig=None, ax=None, name="loss_plot", show=True):
        if fig is None:
            fig, ax = plt.subplots()

        ax.plot(self.epoch_loss, label='Training Loss', color='blue')
        if len(self.epoch_val_loss) == len(self.epoch_loss): ax.plot(self.epoch_val_loss, label='Validation Loss', color='orange')

        # Adding labels and legend
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Losses')
        ax.set_title('Training Loss')

        # add color legend
        legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Validation Loss'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Training Loss'),
        ]

        # Adding legend with custom elements
        plt.legend(handles=legend_elements, loc='upper right')

        if save:
            if not os.path.exists("./images"):
                os.makedirs("./images")
            if name == "loss_plot":
                plt.savefig("./images/loss_plot"+dt.now().strftime('%d-%m-%Y_%H-%M-%S')+".png")
            else:
                plt.savefig("./images/"+name+".png")

        # show updates online without blocking the code
        if show:
            plt.show()
            plt.pause(0.0005)



# DEBUG AND LOG METHODS


    # Method to print the neural network architectural structure
    def print_net(self):
        print("\n"+repr(self)+"\n")
        print("=" * 50)
        print("=" * 15 +" MLP Neural Network "+"=" * 15)
        print("=" * 50 + "\n")
        for l in range(len(self.net)):                              # loop over layers
            Wl, bl = self.net[l]
            print(f"Layer {l} --- Neurons: {bl.shape[0]} --- Activation: {self.layers[l][1]}")
            print(f"W: {Wl.shape} \t b: {bl.shape}\n")
            print(f"W: {Wl}\n")
            print(f"b: {bl}")
            print("=" * 50)
    
    
    
    # Override of default methods
    def __str__(self):
        return f"MLP with {len(self.layers)} layers"

    def __repr__(self):
        return repr(f"MLP NN (Input : {self.num_input} , Layers : {self.layers} , Depht : {len(self.net)} , Loss : {self.loss_name})")

    def __call__(self, X):
        return self.predict(X)

    def __getitem__(self, i):
        return self.net[i]










