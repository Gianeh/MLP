import numpy as np
class MLP:
    def __init__(self, num_input, num_output, hidden_shape = [], activation = "sigmoid", loss = "mse", log_rate = 100):
        # base parameters
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_shape = hidden_shape
        # initialize the network tensor list
        self.net = []
        self.outputs = []
        self.activations = []
        self.init_net()
        # initialize loss and activation functions
        self.loss = self.init_loss(loss)
        self.activation = self.init_activation(activation)
        # NOTE: ideally every layer should have it's own activation function, but for now we'll just use the same for every layer

        self.log_rate = log_rate

    def init_net(self):
        for i in range(len(self.hidden_shape) + 1):
            if i == 0:
                self.net.append((
                    np.random.rand(self.num_input, self.hidden_shape[i]),
                    np.random.randn(self.hidden_shape[i])
                ))
            elif i < len(self.hidden_shape):
                self.net.append((
                    np.random.rand(self.hidden_shape[i-1], self.hidden_shape[i]),
                    np.random.randn(self.hidden_shape[i])
                    ))
            elif i == len(self.hidden_shape):
                self.net.append((
                    np.random.rand(self.hidden_shape[i-1], self.num_output),
                    np.random.randn(self.num_output)
                ))
        self.outputs = [None] * (len(self.hidden_shape)+1)
        self.activations = [None] * (len(self.hidden_shape)+1)

    def init_loss(self, loss):
        if loss == "mse":
            return self.mse
        else:
            raise ValueError("Invalid loss function")
    
    def init_activation(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("Invalid activation function")
        
    # Activation functions:
    
    def sigmoid(self, A, grad=False):
        return 1 / (1.0 + np.exp(-A)) if not grad else self.sigmoid(A) * (1.0 - self.sigmoid(A))
    
    # Loss functions:

    def mse(self, OL, y, grad=False):
        diff = OL - y
        return np.mean(diff * diff) if not grad else (2.0 / y.shape[0]) * diff

    def forward(self, X):
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            if l == 0:
                self.activations[l] = np.matmul(X, Wl) + bl  # first layer: the input is the given data X (broadcast!)
            else:
                self.activations[l] = np.matmul(self.outputs[l - 1], Wl) + bl  # other layers: the input is the output of the layer below (broadcast!)
            self.outputs[l] = self.activation(self.activations[l])  # output of each layer
        return self.outputs, self.activations
    
    def backward(self, X, O, A, d_loss):
        G = [None] * len(self.net)
        for l in range(len(self.net) - 1, -1, -1):
            if l == len(self.net) - 1:
                Delta = self.activation(A[l], grad=True) * d_loss
            else:
                Wl_plus_1, _ = self.net[l + 1]
                Delta = self.activation(A[l], grad=True) * (np.matmul(Delta, Wl_plus_1.T))
            if l > 0:
                Wl_grad = np.matmul(O[l - 1].T, Delta)
            else:
                Wl_grad = np.matmul(X.T, Delta)
            bl_grad = np.sum(Delta, axis=0)
            G[l] = (Wl_grad, bl_grad)
        return G
    
    def update(self, G, lr=0.001):
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            Wl_grad, bl_grad = G[l]
            Wl -= lr * Wl_grad
            bl -= lr * bl_grad
            self.net[l] = (Wl, bl)

    def train(self, X, y, batch_size, epochs=100, lr=0.001):
        # if batch size is 0, train on the whole dataset
        if batch_size == 0:
            batch_size = len(X)
        # if batch size is not a multiple of the dataset size, train on the whole dataset and print a warning
        if len(X) % batch_size != 0:
            print("Warning: batch size is not a multiple of the dataset size. Training on the whole dataset.")
            batch_size = len(X)
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                O, A = self.forward(X[i:i+batch_size])
                d_loss = self.loss(O[-1], y[i:i+batch_size], grad=True)
                G = self.backward(X[i:i+batch_size], O, A, d_loss)
                self.update(G, lr=lr)
            if epoch % self.log_rate == 0:
                print(f"Epoch {epoch} - Loss: {self.loss(O[-1], y[i:i+batch_size])}")
    def predict(self, X):
        O, _ = self.forward(X)
        return O[-1]
    
    def evaluate(self, X, y):
        O, _ = self.forward(X)
        return self.loss(O[-1], y)
    
    def print_net(self):
        print("Network length: {}".format(len(self.net)))
        for l in range(len(self.net)):
            Wl, bl = self.net[l]
            print(f"Layer {l} - W: {Wl.shape} - b: {bl.shape}")
            print(f"W: {Wl}")
            print(f"b: {bl}")
            print("="*30)
    
    # override some default methods
    def __str__(self):
        return f"MLP with {len(self.hidden_shape)} hidden layers"
    
    def __repr__(self):
        return f"MLP({self.num_input}, {self.num_output}, {self.hidden_shape})"
    
    def __call__(self, X):
        return self.predict(X)
    
    def __getitem__(self, i):
        return self.net[i]