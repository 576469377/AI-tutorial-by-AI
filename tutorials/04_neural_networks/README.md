# Neural Networks Introduction

Explore the fundamentals of neural networks and deep learning with visual examples and practical implementations.

## What You'll Learn

- Neural network basics and terminology
- How neurons and layers work
- Activation functions and their purposes
- Training process (forward/backward propagation)
- Building simple neural networks from scratch
- Visualization of network architectures

## Prerequisites

- Completed [Machine Learning Basics](../03_machine_learning/README.md)
- Understanding of linear algebra (vectors, matrices)
- Basic calculus concepts (derivatives)

## Topics Covered

### 1. Neural Network Fundamentals
- What is a neural network?
- Biological inspiration vs artificial implementation
- Perceptron: the building block
- Multi-layer perceptrons (MLPs)

### 2. Key Components
- **Neurons/Nodes**: Processing units
- **Weights and Biases**: Parameters to learn
- **Activation Functions**: Adding non-linearity
- **Layers**: Input, hidden, and output layers

### 3. Training Process
- Forward propagation: Making predictions
- Loss functions: Measuring error
- Backward propagation: Learning from mistakes
- Gradient descent: Optimization algorithm

### 4. Activation Functions
- Sigmoid: For binary classification
- ReLU: Most common in hidden layers
- Tanh: Zero-centered alternative
- Softmax: For multi-class problems

## Network Architecture Examples

```
Simple Neural Network for Binary Classification:

Input Layer    Hidden Layer    Output Layer
    X1  ────────── H1 ──────────── Y
     │   ╲       ╱  │  ╲       ╱
     │    ╲     ╱   │   ╲     ╱
    X2  ────────── H2 ──────────
     │   ╱     ╲   │   ╱     ╲
     │  ╱       ╲  │  ╱       ╲
    X3  ────────── H3 ──────────
```

## Examples

Learn neural networks through code:
- **Notebook**: [04_neural_networks.ipynb](../../notebooks/04_neural_networks.ipynb)
- **Script**: [04_neural_network_examples.py](../../examples/04_neural_network_examples.py)

## Practical Applications

1. **Image Classification**: Recognizing objects in photos
2. **Natural Language Processing**: Understanding text
3. **Recommendation Systems**: Suggesting products/content
4. **Time Series Forecasting**: Predicting future values
5. **Game Playing**: AI that learns to play games

## Common Challenges

- **Overfitting**: Model memorizes training data
- **Vanishing Gradients**: Deep networks struggle to learn
- **Local Minima**: Getting stuck in suboptimal solutions
- **Hyperparameter Tuning**: Choosing the right settings

## Next Steps

- Explore deep learning frameworks (TensorFlow, PyTorch)
- Study convolutional neural networks (CNNs) for images
- Learn about recurrent neural networks (RNNs) for sequences
- Discover advanced architectures (Transformers, GANs)

## Glossary

- **Epoch**: One complete pass through the training data
- **Batch**: Subset of training data processed together
- **Learning Rate**: How fast the model learns (step size)
- **Gradient**: Direction of steepest increase in loss function
- **Backpropagation**: Algorithm for computing gradients