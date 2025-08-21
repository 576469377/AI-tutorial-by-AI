# Neural Networks Introduction

Enter the world of deep learning by mastering neural network fundamentals with visual examples and practical implementations.

## 🎯 Learning Track

**🧠 You are in: Deep Learning Track**  
**⏱️ Estimated time**: 4-6 weeks (this + PyTorch)  
**🎯 Best for**: AI research, computer vision, NLP, advanced AI applications

## What You'll Learn

- Neural network basics and core terminology
- How neurons and layers work mathematically
- Activation functions and their purposes
- Training process (forward/backward propagation)
- Building simple neural networks from scratch
- Visualization of network architectures and decision boundaries

## Prerequisites

- ✅ Completed [AI Fundamentals](../00_ai_fundamentals/README.md)
- ✅ Completed [Python Basics](../01_basics/README.md) 
- ✅ Completed [Data Visualization](../02_data_visualization/README.md)
- 🧮 Understanding of linear algebra (vectors, matrices) - **Essential**
- 🧮 Basic calculus concepts (derivatives) - **Essential**
- 📊 *Optional but helpful*: [Machine Learning Basics](../03_machine_learning/README.md)

## 🎓 Learning Objectives

By the end of this tutorial, you will:
- Understand the fundamental differences between traditional ML and deep learning
- Build neural networks from scratch using NumPy
- Implement and visualize different activation functions
- Train networks using gradient descent and backpropagation
- Recognize when neural networks are the right solution

## 🔍 Deep Learning vs Traditional ML

**Neural Networks excel when:**
- 🧠 Large amounts of data available (>10K+ samples)
- 🧠 Complex patterns in data (images, audio, text)
- 🧠 Non-linear relationships
- 🧠 Feature extraction should be learned automatically
- 🧠 State-of-the-art performance is critical

**Traditional ML is better when:**
- 📊 Small datasets or limited data
- 📊 Need interpretable models
- 📊 Quick training and deployment
- 📊 Tabular/structured data
- 📊 Limited computational resources

*For traditional ML techniques, see [Machine Learning Basics](../03_machine_learning/README.md)*

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

## Mathematical Foundations

### Backpropagation Algorithm

The backpropagation algorithm is the core of neural network training. Here's the mathematical foundation:

#### Forward Pass
For a layer l with weights W^(l) and biases b^(l):
```
z^(l) = W^(l) × a^(l-1) + b^(l)
a^(l) = σ(z^(l))
```
Where σ is the activation function.

#### Backward Pass
The error at the output layer L:
```
δ^(L) = ∇_a C ⊙ σ'(z^(L))
```

For hidden layers:
```
δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))
```

#### Parameter Updates
```
∂C/∂W^(l) = a^(l-1) δ^(l)
∂C/∂b^(l) = δ^(l)

W^(l) := W^(l) - η ∂C/∂W^(l)
b^(l) := b^(l) - η ∂C/∂b^(l)
```

Where η is the learning rate.

### Activation Functions Mathematics

#### Sigmoid Function
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))
```
- Output range: (0, 1)
- Problem: Vanishing gradient for large |x|

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```
- Advantages: Simple, no vanishing gradient for x > 0
- Problem: "Dead neurons" for x ≤ 0

#### Tanh (Hyperbolic Tangent)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```
- Output range: (-1, 1)
- Zero-centered, good for hidden layers

#### Softmax (for multi-class output)
```
softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```
- Converts logits to probabilities
- Sum of outputs = 1

### Loss Functions Mathematics

#### Mean Squared Error (Regression)
```
MSE = (1/n) Σ(y_i - ŷ_i)²
∂MSE/∂ŷ = 2(ŷ - y)/n
```

#### Cross-Entropy Loss (Classification)
For multi-class:
```
CE = -Σ y_i log(ŷ_i)
∂CE/∂ŷ_i = -y_i/ŷ_i
```

For binary classification:
```
BCE = -(y log(ŷ) + (1-y) log(1-ŷ))
```

### Gradient Descent Variants

#### Standard Gradient Descent
```
θ := θ - η ∇J(θ)
```

#### Momentum
```
v := γv + η ∇J(θ)
θ := θ - v
```
- γ: momentum coefficient (typically 0.9)
- Helps accelerate gradients in relevant direction

#### Adam (Adaptive Moment Estimation)
```
m := β₁m + (1-β₁) ∇J(θ)
v := β₂v + (1-β₂) (∇J(θ))²
m̂ := m / (1-β₁^t)
v̂ := v / (1-β₂^t)
θ := θ - η m̂ / (√v̂ + ε)
```
- Combines benefits of momentum and RMSprop
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (typical values)

### Universal Approximation Theorem

**Theorem**: A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Rⁿ, given appropriate activation functions.

**Implications**:
- Neural networks are theoretically capable of learning any function
- In practice, finding the right weights is the challenge
- Deeper networks often learn more efficiently than wider ones

### Regularization Mathematics

#### L1 Regularization (Lasso)
```
Loss_regularized = Loss_original + λ Σ|w_i|
```
- Promotes sparsity (many weights become exactly 0)
- Feature selection effect

#### L2 Regularization (Ridge)
```
Loss_regularized = Loss_original + λ Σw_i²
```
- Penalizes large weights
- Prevents overfitting by encouraging smaller weights

#### Dropout
During training, randomly set neurons to 0 with probability p:
```
h_dropout = h ⊙ mask, where mask ~ Bernoulli(1-p)
```
At test time, scale outputs: h_test = (1-p) × h

### Information Theory in Neural Networks

#### Entropy
Measures uncertainty in a probability distribution:
```
H(X) = -Σ P(x) log P(x)
```

#### Cross-Entropy
Measures difference between two probability distributions:
```
H(P,Q) = -Σ P(x) log Q(x)
```
Used as loss function in classification.

#### Mutual Information
Measures dependence between variables:
```
I(X;Y) = H(X) - H(X|Y)
```
Important for understanding what networks learn.

## Advanced Concepts

### Convolutional Neural Networks (CNNs)

#### Convolution Operation
For 2D convolution:
```
(f * g)[m,n] = ΣΣ f[i,j] × g[m-i, n-j]
```

#### Key Parameters
- **Kernel size**: Size of the filter
- **Stride**: Step size for moving the filter
- **Padding**: Adding zeros around input
- **Output size**: (Input + 2×Padding - Kernel) / Stride + 1

### Recurrent Neural Networks (RNNs)

#### Basic RNN
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

#### LSTM Gates
- **Forget gate**: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
- **Input gate**: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
- **Candidate values**: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
- **Cell state**: C_t = f_t * C_{t-1} + i_t * C̃_t
- **Output gate**: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
- **Hidden state**: h_t = o_t * tanh(C_t)

## Practical Implementation Tips

### Numerical Stability
- Use log-softmax instead of softmax when possible
- Clip gradients to prevent explosion: ||g|| ≤ threshold
- Use batch normalization to stabilize training

### Weight Initialization
- **Xavier/Glorot**: Variance = 2/(fan_in + fan_out)
- **He initialization**: Variance = 2/fan_in (for ReLU)
- **Random normal**: μ = 0, σ = 0.01

### Learning Rate Scheduling
- **Step decay**: Reduce LR by factor every N epochs
- **Exponential decay**: LR = LR₀ × γ^epoch
- **Cosine annealing**: LR = LR_min + (LR_max - LR_min) × (1 + cos(π × epoch/T))/2

## 🚀 Next Steps: Continue Deep Learning Track

### 📍 Your Current Progress
🧠 **Deep Learning Track**: Neural Networks ✅ → PyTorch → Advanced AI

### Immediate Next Step  
**→ [PyTorch Deep Learning](../05_pytorch/README.md)** - Essential for modern deep learning
- Learn the industry-standard deep learning framework
- Build practical neural networks with GPU acceleration
- Implement CNNs, RNNs, and advanced architectures

### Future Learning Path
After PyTorch, you can:
- **🚀 [Large Language Models](../06_large_language_models/README.md)** - Advanced AI track
- **📊 [Machine Learning Basics](../03_machine_learning/README.md)** - Broaden your ML knowledge

### Alternative Paths
- **Focus on Computer Vision**: Study CNNs in PyTorch tutorial
- **Focus on NLP**: Study RNNs in PyTorch, then move to LLMs
- **Research Path**: Master all tracks for comprehensive understanding

### 🎯 Skills Gained So Far
- ✅ Neural network fundamentals
- ✅ Mathematical understanding of deep learning
- ✅ Forward and backward propagation
- ✅ Architecture design principles

### 📚 Additional Practice
- Implement networks for different domains (vision, NLP, tabular data)
- Experiment with different activation functions and architectures
- Visualize learned representations and decision boundaries
- Study specific architectures (ResNet, LSTM, Transformer basics)

## Glossary

- **Epoch**: One complete pass through the training data
- **Batch**: Subset of training data processed together
- **Learning Rate**: How fast the model learns (step size)
- **Gradient**: Direction of steepest increase in loss function
- **Backpropagation**: Algorithm for computing gradients