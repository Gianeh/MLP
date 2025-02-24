# Multilayer Perceptron (MLP) Proof-of-Concept Library

This repository contains a lightweight, proof-of-concept implementation of a Multilayer Perceptron (MLP) neural network library developed as part of an academic project at the University of Siena (Academic Year 2023-2024).

---

## Overview

This MLP library provides a flexible implementation of feedforward neural networks (MLPs), enabling you to design, train, and evaluate networks for a variety of tasks. Key features include:

- **Customizable Network Architectures:**  
  Define your own network by specifying the number of layers, neurons per layer, and activation functions. The library supports classic choices like sigmoid, ReLU, Leaky ReLU, and softmax.

- **Training and Inference:**  
  Perform forward propagation and backpropagation with support for both full-batch and mini-batch training. The library comes with built-in options for early stopping and validation monitoring.

- **Optimizers:**  
  Implement multiple learning rate strategies and optimizers such as Basic Adaptive Learning Rate, RPROP, and ADAM to update network weights and biases efficiently.

- **Evaluation Metrics:**  
  Automatically compute performance measures—including accuracy, precision, recall, F1-score, and confusion matrices—making it easier to assess your model on both classification and regression tasks.

- **Experiments on Standard Datasets:**  
  The library has been tested on classic problems like XOR, as well as on standard datasets such as Boston Housing (regression), Wine (classification), and MNIST (handwritten digit recognition). Detailed experimental results and the best-performing configurations are described in the report.

---

## Key Details from the Report

- **MLP Fundamentals:**  
  The report provides a comprehensive description of the MLP model as a Directed Acyclic Graph (DAG) of interconnected neurons organized in layers. While it includes in-depth mathematical formulas for forward propagation, backpropagation, and gradient calculation, the library abstracts these details for practical usage.

- **Activation and Loss Functions:**  
  Various activation functions (e.g., sigmoid, ReLU, softmax) and loss functions (e.g., MSE, cross-entropy) are discussed in the report. These functions are implemented in the library as selectable options for different types of tasks.

- **Learning Rate Optimizers:**  
  In addition to standard gradient descent, the report covers advanced techniques such as RPROP and ADAM, which are incorporated into the library to ensure stable and efficient learning.

- **Experimental Validation:**  
  The report details experiments on multiple datasets. For instance, the MNIST experiments utilized a grid search over network hyperparameters (architecture: 784 → 900 → 1000 → 200 → 10 with ADAM optimizer) to achieve high accuracy and robust performance metrics. While the math behind these experiments is extensive, you can use the provided MLP class to replicate or extend these experiments without needing to reimplement the underlying theory.

For a deeper dive into these topics, please refer to the included `report.pdf`.

---

## Installation

Clone the repository and ensure you have Python 3.x installed:

```bash
git clone https://github.com/Gianeh/MLP.git
cd MLP
```

Install the required dependencies using pip:

```bash
pip install numpy matplotlib
```

*(Additional libraries such as Pickle or Deeplake may be required if you plan to use extended functionalities.)*

---

## Usage

The library is designed as a proof-of-concept. You can import the MLP class and configure your network parameters as needed. Here’s a simple example:

```python
from mlp import MLP  # Import the main MLP class

# Define an MLP for an example task (e.g., MNIST)
# For MNIST, the input size is 784 (28x28 flattened), and the output size is 10 (one per digit).
model = MLP(
    layer_sizes=[784, 900, 1000, 200, 10],
    activation_functions=['sigmoid', 'sigmoid', 'sigmoid', 'softmax'],
    optimizer='adam',
    learning_rate=0.0001
)

# Train the model on your dataset (replace X_train, Y_train with your training data)
model.train(X_train, Y_train, epochs=50, batch_size=3000, validation_data=(X_val, Y_val))

# Evaluate the model on test data
metrics = model.evaluate(X_test, Y_test)
print(metrics)
```

Refer to the code comments for additional configuration options such as early stopping and detailed evaluation metrics.

---

## Experiments

The library has been applied to a variety of tasks:
- **XOR Problem:** Demonstrates the capability to learn non-linear patterns.
- **Boston Housing:** A regression task predicting house prices based on 13 features.
- **Wine Classification:** A multiclass classification task using chemical properties.
- **MNIST Digit Recognition:** The library achieved high performance using a deep network architecture tuned via grid search.

Detailed experimental setups and results are provided in the report.

---

## Report

For a comprehensive description of the network architecture, training algorithms, activation and loss functions, and experimental results, please refer to the project report (`report.pdf`). This document offers deep insights into the theoretical foundations and experimental validation behind the library.

---

## Acknowledgments

This project is the result of collaborative academic work at the University of Siena, developed by Kevin Gagliano, Pietro Pianigiani, and Fabrizio Benvenuti
