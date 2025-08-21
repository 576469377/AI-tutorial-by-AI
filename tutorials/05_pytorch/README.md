# PyTorch Deep Learning Tutorial

Master deep learning with PyTorch, one of the most popular and flexible deep learning frameworks. This tutorial covers everything from basic tensor operations to building and training complex neural networks.

## What You'll Learn

- PyTorch fundamentals: tensors, autograd, and computational graphs
- Building neural networks with nn.Module
- Training loops and optimization
- Convolutional Neural Networks (CNNs) for image processing
- Recurrent Neural Networks (RNNs) for sequence data
- Advanced techniques: transfer learning, regularization, and model deployment

## Prerequisites

- Completed [Neural Networks Introduction](../04_neural_networks/README.md)
- Understanding of Python and NumPy
- Basic knowledge of linear algebra and calculus
- Familiarity with machine learning concepts

## Topics Covered

### 1. PyTorch Fundamentals

#### What is PyTorch?
PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides:
- **Dynamic computational graphs**: Build graphs on-the-fly
- **Automatic differentiation**: Compute gradients automatically
- **GPU acceleration**: Seamless CUDA integration
- **Python-first design**: Intuitive and flexible API

#### Tensors: The Building Blocks
Tensors are multi-dimensional arrays, similar to NumPy arrays but with additional capabilities:

```python
import torch
import numpy as np

# Creating tensors
x = torch.tensor([1, 2, 3, 4, 5])  # From list
y = torch.from_numpy(np.array([1, 2, 3]))  # From NumPy
z = torch.zeros(3, 4)  # Zeros tensor
w = torch.randn(2, 3)  # Random tensor
```

#### Mathematical Operations
```python
# Basic operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
c = a + b  # Addition
d = a * b  # Multiplication
e = torch.dot(a, b)  # Dot product

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = torch.mm(A, B)  # Matrix multiplication
```

### 2. Automatic Differentiation (Autograd)

PyTorch's autograd system automatically computes gradients for backpropagation:

```python
# Enable gradient computation
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1

# Compute gradient
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 2(2) + 3 = 7
```

#### Mathematical Foundation
For a function f(x), the gradient ∇f gives the direction of steepest increase:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

Chain rule for composite functions:
```
If z = f(g(x)), then dz/dx = (dz/dg) × (dg/dx)
```

### 3. Building Neural Networks

#### The nn.Module Framework
All neural networks in PyTorch inherit from `nn.Module`:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleNN(784, 128, 10)  # MNIST classifier
```

#### Common Layer Types
- **nn.Linear**: Fully connected layer
- **nn.Conv2d**: 2D convolutional layer
- **nn.LSTM**: Long Short-Term Memory layer
- **nn.Dropout**: Regularization layer
- **nn.BatchNorm1d/2d**: Batch normalization

### 4. Training Neural Networks

#### Complete Training Loop
```python
import torch.optim as optim

# Setup
model = SimpleNN(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

#### Mathematical Foundation: Gradient Descent
Update rule: θ = θ - α∇J(θ)
- θ: parameters
- α: learning rate
- ∇J(θ): gradient of loss function

### 5. Convolutional Neural Networks (CNNs)

CNNs are designed for processing grid-like data such as images:

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### Mathematical Foundation: Convolution Operation
For 2D convolution:
```
(f * g)[m,n] = ∑∑ f[i,j] × g[m-i, n-j]
```

Key concepts:
- **Filters/Kernels**: Detect features like edges, corners
- **Stride**: Step size for filter movement
- **Padding**: Add zeros around input
- **Pooling**: Reduce spatial dimensions

### 6. Recurrent Neural Networks (RNNs)

RNNs process sequential data by maintaining hidden states:

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out
```

#### Mathematical Foundation: LSTM Gates
LSTM uses gates to control information flow:

```
Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
```

Where σ is the sigmoid function.

### 7. Advanced Techniques

#### Transfer Learning
Use pre-trained models as feature extractors:

```python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# Freeze parameters
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
```

#### Regularization Techniques

**Dropout**: Randomly zero some neurons during training
```python
self.dropout = nn.Dropout(0.5)
```

**Batch Normalization**: Normalize layer inputs
```python
self.bn1 = nn.BatchNorm2d(64)
```

**Weight Decay**: L2 regularization in optimizer
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

#### Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# In training loop
scheduler.step()
```

### 8. Model Evaluation and Debugging

#### Evaluation Mode
```python
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient computation
    for data, target in test_loader:
        output = model(data)
        # Compute metrics
```

#### Common Debugging Techniques
- **Print tensor shapes**: Ensure dimensions match
- **Check gradients**: Verify gradient flow
- **Visualize training curves**: Plot loss and accuracy
- **Monitor GPU memory**: Avoid out-of-memory errors

### 9. GPU Acceleration

#### Moving to GPU
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data to GPU
model = model.to(device)
data = data.to(device)
target = target.to(device)
```

#### Mixed Precision Training
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# In training loop
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 10. Model Deployment

#### Saving and Loading Models
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleNN(784, 128, 10)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

#### TorchScript for Production
```python
# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Load and use
loaded_model = torch.jit.load('model_scripted.pt')
```

## Examples

Explore PyTorch in action:
- **Notebook**: [05_pytorch_tutorial.ipynb](../../notebooks/05_pytorch_tutorial.ipynb)
- **Script**: [05_pytorch_examples.py](../../examples/05_pytorch_examples.py)

## Sample Projects

1. **MNIST Digit Classification** - CNN implementation
2. **CIFAR-10 Image Recognition** - Advanced CNN with data augmentation
3. **Text Sentiment Analysis** - RNN/LSTM for natural language processing
4. **Transfer Learning** - Fine-tuning pre-trained models
5. **Generative Adversarial Network (GAN)** - Generate synthetic images

## Performance Tips

### Memory Optimization
- Use `torch.no_grad()` for inference
- Delete unnecessary variables with `del`
- Use gradient checkpointing for large models
- Process data in smaller batches

### Speed Optimization
- Use DataLoader with multiple workers
- Pin memory for faster GPU transfer
- Use compiled models with TorchScript
- Optimize data preprocessing pipeline

## Common Pitfalls and Solutions

### Gradient Issues
- **Vanishing gradients**: Use ReLU, batch normalization, residual connections
- **Exploding gradients**: Use gradient clipping
- **No gradients**: Check `requires_grad=True`

### Training Issues
- **Overfitting**: Use regularization, more data, early stopping
- **Underfitting**: Increase model capacity, reduce regularization
- **Slow training**: Increase learning rate, use GPU, optimize data loading

### Memory Issues
- **Out of memory**: Reduce batch size, use gradient accumulation
- **Memory leaks**: Use `del`, avoid creating unnecessary graphs

## Mathematical Formulations

### Backpropagation Algorithm
For a neural network with L layers:

1. **Forward pass**: Compute activations
   ```
   a^(l) = σ(W^(l)a^(l-1) + b^(l))
   ```

2. **Backward pass**: Compute gradients
   ```
   δ^(L) = ∇_a C ⊙ σ'(z^(L))
   δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))
   ```

3. **Update parameters**:
   ```
   W^(l) := W^(l) - η ∂C/∂W^(l)
   b^(l) := b^(l) - η ∂C/∂b^(l)
   ```

### Loss Functions

**Mean Squared Error (Regression)**:
```
MSE = (1/n) ∑(y_i - ŷ_i)²
```

**Cross-Entropy (Classification)**:
```
CE = -∑ y_i log(ŷ_i)
```

**Binary Cross-Entropy**:
```
BCE = -(y log(ŷ) + (1-y) log(1-ŷ))
```

## Next Steps

Ready for advanced deep learning?
- Explore computer vision with torchvision
- Study natural language processing with torchtext
- Learn about generative models (GANs, VAEs)
- Experiment with reinforcement learning
- Contribute to open-source PyTorch projects

## Additional Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)

### Books
- "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann
- "Programming PyTorch for Deep Learning" by Ian Pointer

### Courses
- Fast.ai: Practical Deep Learning for Coders
- CS231n: Convolutional Neural Networks for Visual Recognition
- CS224n: Natural Language Processing with Deep Learning

### Research Papers
- "Attention Is All You Need" (Transformers)
- "ResNet: Deep Residual Learning for Image Recognition"
- "LSTM: Long Short-Term Memory Networks"

---

**Remember**: PyTorch's strength lies in its flexibility and ease of debugging. Start with simple examples and gradually build complexity. The dynamic nature of PyTorch makes it perfect for research and experimentation!

Happy deep learning! 🔥🧠