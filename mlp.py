#class MLP is designed to process row vector input and target, in general MLP takes as input a matrix where each row is one train sample
#and as target a matrix where each row is one target vector

#internally every calculation is treated according to the conventional algebraic rules (reshaping of the input arguments passed to the net)
#where each input is a column vector (forming a matrix for multiple samples) that produce an output that is a column vector too
#in order to compute the loss, error calculation is the difference between two column vectors (online learning) or between two matrices (mini-batch or full-batch)
#so also the targets need to be treated as column vectors possibly forming a matrix for (mini/full-batch)


import numpy as np
import pickle

class MLP:
    def __init__(self, num_input, layers =[], loss = "mse", log_rate = 100):

        self.num_input = num_input
        self.layers = layers
        self.net = []
        self.outputs = []
        self.activations = []
        self.init_net()
        self.loss = self.init_loss(loss)
        self.activation_functions = self.init_activation_functions(layers)
        self.log_rate = log_rate
        self.loss_name = loss



#init_net initialize according to the normal distribution all the matrices weights and all the bias vectors according to the standard algebraic rules
#furthermore at the end of the loop over all the layers, it initializes the lists of outputs and activations of the net, for each layer as None value

    def init_net(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.net.append((
                    np.random.randn(self.layers[i][0], self.num_input),
                    np.random.randn(self.layers[i][0],1)
                ))
            elif i < len(self.layers):      #in this case else is equivalent to elif(condition)
                self.net.append((
                    np.random.randn(self.layers[i][0], self.layers[i-1][0]),
                    np.random.randn(self.layers[i][0],1)
                ))
        self.outputs = [None] * (len(self.layers))
        self.activations = [None] * (len(self.layers))


    def init_loss(self, loss):
        if loss == "mse":
            return self.mse
        elif loss == "categorical_crossentropy":
            return self.categorical_crossentropy
        else:
            raise ValueError("Invalid loss function")


    def init_activation_functions(self, layers):
        activation_functions = []
        for layer in layers:
            if layer[1] == "sigmoid":
                activation_functions.append(self.sigmoid)
            elif layer[1] == "relu":
                activation_functions.append(self.relu)
            elif layer[1] == "leaky_relu":
                activation_functions.append(self.leaky_relu)
            elif layer[1] == "tanh":
                activation_functions.append(self.tanh)
            elif layer[1] == "softmax":
                activation_functions.append(self.softmax)
            elif layer[1] == "linear":
                activation_functions.append(self.linear)
            else:
                raise ValueError("Invalid activation function")
        return activation_functions



#The sigmoid activation function is an element-wise function that takes as input the activation column vector (for a batch is a matrix), where:
#if grad=False return a column vector too that has for each component the sigmoidal computation of the corresponding argument component
#if grad=True return a column vector where the component i is sigmoid(A[i]) * (1.0-sigmoid(A[i]))
#Remark: input A and output have always the same dimensions

    def sigmoid(self, A, grad=False):
        if not grad:
            return 1.0 / (1.0 + np.exp(-A))
        else:
            return self.sigmoid(A) * (1.0 - self.sigmoid(A))


#Relu activation function initialize with all zeros a tensor called output that has the same dimensions as the activation tensor argument (A)
#if grad=False return a tensor where each component is computed by the relu scalar function (max(0.0, x))
#if grad=True return a tensor where each component is 0 or 1 (derivative of relu is 0 for negative activation value and 1 for null or positive value)

    def relu(self, A, grad=False):
        output = np.zeros(A.shape)
        if not grad:
            for col in range(0, output.shape[1]):                   
                for row in range(0, output.shape[0]):
                    output[row][col] = max(0.0, A[row][col])
            return output
        else:
            for col in range(0, output.shape[1]):
                for row in range(0, output.shape[0]):
                    output[row][col] = 1 if (A[row][col] > 0) else 0
            return output


#Leaky Relu activation function initialize with all zeros a tensor called output that has the same dimensions as the activation tensor argument (A)
#if grad=False return a tensor where each component is computed by the leaky relu scalar function (y=f(x) -> y=x if x>=0 and y=mx if x<0 where m is negative_slope variable)
#if grad=True return a tensor where each component is neg_slope or 1 (derivative of leaky relu is neg_slope for negative activation value and 1 for null or positive value)
    
    def leaky_relu(self, A, neg_slope=0.01 ,grad=False):
        output = np.zeros(A.shape)
        if not grad:
            for col in range(0, output.shape[1]):
                for row in range(0, output.shape[0]):
                    output[row][col] = A[row][col] if (A[row][col] >= 0) else (A[row][col] * neg_slope)
            return output
        else:
            for col in range(0, output.shape[1]):
                for row in range(0, output.shape[0]):
                    output[row][col] = 1 if (A[row][col] >= 0) else neg_slope
            return output


#The hyperbolic tangent activation function is an element-wise function that takes as input the activation column vector (for batch is a matrix), where:
#if grad=False return a column vector too that has for each component the tanh computation of the corresponding argument component
#if grad=True return a column vector where each component is ->  1 - (tanh(a)^2)
   
    def tanh(self, A, grad=False):
        if not grad:
            return (np.exp(A) - np.exp(-A)) / (np.exp(A) + np.exp(-A))
        else:
            return (1.0 - self.tanh(A)**2)


#The softmax activation function is an activation where each output component is function of all the input arguments
#it is assumed that this activation is used only in output layer in combination with categorical categorical crossentropy loss
#given this assumption, it is not necessary to calculate the derivative of the activation function since it is integrated into the
#derivative of the loss function (not included in the loss code, but taken into account in the backward and train method)
   
    def softmax(self, A):
        output = np.zeros(A.shape)
        for col in range(0, output.shape[1]):
            for row in range(0, output.shape[0]):
                output[row][col] = np.exp(A[row][col])
            normalizer = np.sum(output[: , col], axis=0)
            output[:, col] = (output[: , col] / normalizer)
        return output


#Linear activation function
    
    def linear(self, A, grad=False):
        if not grad:
            return A
        else:
            return np.ones(A.shape)


#The mse loss function calculates the error tensor (matrix if mini/full-batch and column vector if online), where:
#if grad:False return always a scalar value (vector with one element) that is the mean over all the dimension of the squared error tensor
#(mean over all components == err_squared.mean(axis=0) -> row vector; than mean over all samples == err_squared.mean(axis=0) -> scalar)
#where err * err is an element wise multiplication so has the same dimensions of err column vector
#if grad=True return always a tensor that has the same shape as OL(predictions tensor) and Y(targets tensor) and where each component
#is normalized by mean (1/Y.shape[0] where Y.shape[0] is the dimensionality of each target)

    def mse(self, OL, Y, grad=False):
        err = OL - Y
        if not grad:
            return np.mean(err * err)
        else:
            return (2.0 / Y.shape[0]) * err


#The categorical crossentropy loss assume 1-hot encoded target for classification task.
#After limiting the output (OL) in a continuous set (0,1) with extrema excluded, the definition of crossentropy (-np.sum(...))
#is applied which returns a row vector where each element refers to a training sample.
#The average of the losses over all the samples is therefore returned
#Remark: note that in this implementation the natural logarithm is used
#***Remark*** : It is not necessary to define the derivative of the loss since, for the same reasons as the softmax activation, it is assumed that softmax and categorical crossentropy
#are used exclusively together, and therefore the derivative of the loss is taken into account in the calculation of the output layer delta in the backward method (and train method too)
    
    def categorical_crossentropy(self, OL, y):
        epsilon = 1e-15  # Small constant to avoid numerical instability
        clipped_OL = np.clip(OL, epsilon, 1 - epsilon)  # Clip values to avoid log(0) or log(1)
        loss = -np.sum(y * np.log10(clipped_OL), axis=0)  #natural log is used, for base 10 -> np.log10
        loss = np.mean(loss)
        return loss


#forward method for each layer compute the activation and the output tensors and collects them in two separated lists

    def forward(self, X):
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            if l == 0:
                self.activations[l] = np.matmul(Wl, X) + bl
            else:
                self.activations[l] = np.matmul(Wl, self.outputs[l - 1]) + bl
            self.outputs[l] = self.activation_functions[l](self.activations[l])
        return self.outputs, self.activations



#backward method for each layer compute the corresponding delta tensor (matrix for full/mini-batch and column vector for online) and uses it
#for the calculation of the gradient tensor (matrix for weights and column vector for biases)
#the product in Delta calculation is always element wise (implies that both argument tensors have the same shape)
#in the calcuation of the gradient of the loss w.r.t. the weights it is possible to demonstrate that aggregation is intrinsic in the matrix by matrix multiplication
#in the calculation of the loss gradient w.r.t. biases a sum reduction is needed over the samples dimension (y==axis=1)
#at the end of each loop iteration (through layers) matrix (weights) and vector (biases) gradients are collected in a list of tuples G[]
#***Remark*** : extra argument Y wad added to simplify the implementation of categorical crossentropy and softmax and not have to explicitly calculate the derivatives of these functions

    def backward(self, X, O, A, loss_derivative, Y):
        L = len(self.net)
        G = [None] * L
        for l in range(L-1, -1, -1):
            if l == L-1:
                if self.layers[l][1] == "softmax":
                    Delta = O[-1] - Y
                else:
                    Delta = self.activation_functions[l](A[l], grad=True) * loss_derivative
            else:
                Wl_plus_1, _ = self.net[l + 1]
                Delta = self.activation_functions[l](A[l], grad=True) * (np.matmul(Wl_plus_1.T, Delta))
            if l > 0:
                Wl_grad = np.matmul(Delta, O[l - 1].T)
            else:
                Wl_grad = np.matmul(Delta, X.T)

            bl_grad = np.sum(Delta, axis=1).reshape(self.net[l][1].shape[0], 1)
            G[l] = (Wl_grad, bl_grad)
        return G



    def update(self, G, lr=0.001):
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            Wl_grad, bl_grad = G[l]
            Wl -= lr * Wl_grad
            bl -= lr * bl_grad
            self.net[l] = (Wl, bl)


#In train method X and Y inputs are transposed in order to process row-element inputs and targets as in the algebraic rule fashion

    def train(self, X, Y, batch_size=0, epochs=100, lr=0.001):
        X=X.T
        Y=Y.T
        # if batch size is 0, train on the whole dataset
        if batch_size == 0:
            batch_size = len(X.T)
        # if batch size is not a multiple of the dataset size, train on the whole dataset and print a warning
        if len(X.T) % batch_size != 0:
            print("Warning: batch size is not a multiple of the dataset size. Training on the whole dataset.")
            batch_size = len(X.T)
        for epoch in range(epochs):
            for i in range(0, len(X.T), batch_size):
                O, A = self.forward(X[:,i:i+batch_size])
                if self.layers[-1][1] == "softmax":
                    G = self.backward(X[:,i:i+batch_size], O, A, loss_derivative = 0, Y = Y)
                else:
                    loss_derivative = self.loss(O[-1], Y[:,i:i+batch_size], grad = True)
                    G = self.backward(X[:,i:i+batch_size], O, A, loss_derivative = loss_derivative, Y = 0)
                self.update(G, lr=lr)
                if epoch % self.log_rate == 0:
                    print(f"Epoch {epoch} - Batch {int(i/batch_size)} - Loss: {self.loss(O[-1], Y[:,i:i+batch_size])}")



    def predict(self, X):
        if(X.ndim == 1):
            X = X.reshape(X.shape[0],1)     #reshaping as a column vector (when X is a row vector == single input)
        else:
            X=X.T                           #transposing the input matrix (when X is a matrix with row elements == multiple inputs)
        O, _ = self.forward(X)
        return O[-1]


    def evaluate(self, X, Y):
        if(X.ndim == 1):
            X=X.reshape(X.shape[0],1)
            Y=Y.reshape(Y.shape[0],1)
        else:
            X=X.T
            Y=Y.T
        O, _ = self.forward(X)
        return self.loss(O[-1], Y)


    def save_model(self, model_name = "model"):
        with open(model_name+".pkl", "wb") as file:
            pickle.dump(self.net, file)
        print(f"Model saved as {model_name}.pkl")


    def load_model(self, model_name = "model"):
        with open(model_name+".pkl", "rb") as file:
            self.net = pickle.load(file)
        print(f"Model loaded from {model_name}.npz")


    def print_net(self):
        print("\n"+repr(self)+"\n")
        print("=" * 50)
        print("=" * 15 +" MLP Neural Network "+"=" * 15)
        print("=" * 50 + "\n")
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            print(f"Layer {l} --- Neurons: {bl.shape[0]} --- Activation: {self.layers[l][1]}")
            print(f"W: {Wl.shape} \t b: {bl.shape}\n")
            print(f"W: {Wl}\n")
            print(f"b: {bl}")
            print("=" * 50)



    # override some default methods
    def __str__(self):
        return f"MLP with {len(self.layers)} layers"

    def __repr__(self):
        return repr(f"MLP NN (Input : {self.num_input} , Layers : {self.layers} , Depht : {len(self.net)} , Loss : {self.loss_name})")

    def __call__(self, X):
        return self.predict(X)

    def __getitem__(self, i):
        return self.net[i]











