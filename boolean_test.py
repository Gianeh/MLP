import numpy as np
from neural_net import MLP
from matplotlib import pyplot as plt

# Genera un dataset per la funzione OR a tre input
data = []
for i in range(1000):
    x1 = np.random.randint(0, 2)
    x2 = np.random.randint(0, 2)
    x3 = np.random.randint(0, 2)
    # add noise
    x1 = np.random.normal(x1, 0.01)
    x2 = np.random.normal(x2, 0.01)
    x3 = np.random.normal(x3, 0.01)
    data.append([x1, x2, x3, int(round(x1) or round(x2) or round(x3))])  # OR function with three inputs
data = np.array(data)

# Visualizza il dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for l in data:
    ax.scatter(l[0], l[1], l[2], c='r' if l[3] == 0 else 'b', marker='x')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

plt.savefig('or3d_plot.png')

# Divide il dataset in set di addestramento e test
train_data = data[:800, :3]
train_target = data[:800, 3].reshape(-1, 1)
test_data = data[800:, :3]
test_target = data[800:, 3].reshape(-1, 1)

# Crea la rete neurale
net = MLP(3, 1, [5], activation="sigmoid", loss="mse", log_rate=5)

# Stampa la struttura della rete
net.print_net()

# Addestra la rete
net.train(train_data, train_target, batch_size=100, epochs=1000, lr=0.1)

# Testa la rete
O = net.predict(np.array([1, 1, 0]))
print("Output per (1, 1, 0):", O[-1])

# Valuta la rete
# test_loss = net.evaluate(test_data, test_target)
# print("Test Loss:", test_loss)
