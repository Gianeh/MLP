import sys
sys.path.append("../")
from mlp import MLP
import numpy as np

# The first experiment with the newly implemented MLP class
# The XOR problem - a single perceptron is not enough to solve it

data_X = np.array([[0.0342810297940277, 0.861838599030015],
                   [0.070405260144561, 0.920827411566111],
                   [0.90724891133908, 0.100338207589913],
                   [0.872472550285269, -0.156926574145951],
                   [0.932225291072776, 1.14295137982601],
                   [0.939592767308627, 1.07677695428515],
                   [-0.0252777467620643, -0.00379700873214028],
                   [0.00261942822037272, 0.0261750203411463]], dtype=np.float32)  # 8x2

data_y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32).reshape((8, 1))  # 8x1


mlp = MLP(2, [[3, "sigmoid"], [3, "relu"],[1, "sigmoid"]], loss="binary_crossentropy", log_rate=50)
mlp.train(data_X,data_y, batch_size=2, epochs=1000, lr=0.05, plot=True)
mlp.plot()
#print('\n\nPrediction: ', mlp.predict(np.array([0.0456, 1.0365])), '\n')


x=np.array([[0.9645, 0.8549], [0.0124, 1.0245]])
y=np.array([[0.12345],[0.9456]])
prediction= mlp.predict(x)
same_prediction=mlp(x)
eval= mlp.evaluate(x,y)
print(f'\nPrediction : {prediction}\nTarget : {y}\nLoss evaluation: {eval}')
correct = np.sum(np.round(prediction) == y)
#mlp.print_net()
