"""
PyTorch Deep Learning Examples
This module demonstrates fundamental PyTorch concepts and implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def pytorch_fundamentals():
    """Demonstrate basic PyTorch tensor operations"""
    print("=== PyTorch Fundamentals ===")
    
    # Creating tensors
    print("1. Creating Tensors:")
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {x}")
    
    y = torch.zeros(3, 4)
    print(f"Zeros tensor shape {y.shape}:\n{y}")
    
    z = torch.randn(2, 3)
    print(f"Random tensor:\n{z}")
    
    # Mathematical operations
    print("\n2. Mathematical Operations:")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"dot product = {torch.dot(a, b)}")
    
    # Matrix operations
    print("\n3. Matrix Operations:")
    A = torch.randn(3, 4)
    B = torch.randn(4, 2)
    C = torch.mm(A, B)
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"Matrix multiplication C = A @ B, shape: {C.shape}")
    
    # GPU operations (if available)
    if torch.cuda.is_available():
        print("\n4. GPU Operations:")
        gpu_tensor = torch.tensor([1, 2, 3]).cuda()
        print(f"GPU tensor: {gpu_tensor}")
        print(f"Device: {gpu_tensor.device}")
    else:
        print("\n4. GPU not available, using CPU")

def autograd_demo():
    """Demonstrate automatic differentiation"""
    print("\n=== Automatic Differentiation (Autograd) ===")
    
    # Basic gradient computation
    print("1. Basic Gradient Computation:")
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2 + 3*x + 1
    
    print(f"x = {x.item()}")
    print(f"y = x² + 3x + 1 = {y.item()}")
    
    y.backward()
    print(f"dy/dx = 2x + 3 = {x.grad.item()}")
    print(f"At x=2: dy/dx = 2(2) + 3 = {2*2 + 3}")
    
    # Multiple variables
    print("\n2. Multiple Variables:")
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    z = x**2 + y**3 + x*y
    
    z.backward()
    print(f"z = x² + y³ + xy")
    print(f"∂z/∂x = 2x + y = {x.grad.item()}")
    print(f"∂z/∂y = 3y² + x = {y.grad.item()}")
    
    # Chain rule demonstration
    print("\n3. Chain Rule:")
    x = torch.tensor([1.0], requires_grad=True)
    u = x**2
    v = u + 1
    z = v**2
    
    z.backward()
    print(f"z = (x² + 1)²")
    print(f"dz/dx using chain rule = {x.grad.item()}")
    # Manual calculation: dz/dx = 2(x² + 1) * 2x = 4x(x² + 1)
    manual_grad = 4 * 1 * (1**2 + 1)
    print(f"Manual calculation: 4x(x² + 1) = {manual_grad}")

class SimpleNN(nn.Module):
    """Simple feedforward neural network"""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def neural_network_classification():
    """Demonstrate neural network for classification"""
    print("\n=== Neural Network Classification ===")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=3, random_state=42)
    
    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleNN(input_size=20, hidden_size=64, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model architecture:\n{model}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Training loop
    model.train()
    train_losses = []
    
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Confusion matrix-like visualization
    plt.subplot(1, 2, 2)
    predicted_np = predicted.numpy()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predicted_np)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('pytorch_classification_results.png', dpi=150, bbox_inches='tight')
    print("Classification results saved as 'pytorch_classification_results.png'")

class CNN(nn.Module):
    """Convolutional Neural Network for image-like data"""
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assuming 28x28 input -> 7x7 after pooling
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def cnn_demo():
    """Demonstrate CNN with synthetic image data"""
    print("\n=== Convolutional Neural Network Demo ===")
    
    # Create synthetic image data (simulating MNIST-like data)
    num_samples = 1000
    X = torch.randn(num_samples, 1, 28, 28)  # Grayscale 28x28 images
    y = torch.randint(0, 10, (num_samples,))  # 10 classes
    
    # Split data
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = CNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"CNN Architecture:\n{model}")
    print(f"Input shape: {X_train.shape}")
    print(f"Output classes: 10")
    
    # Training loop (shortened for demo)
    model.train()
    num_epochs = 20
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualize some filters
    plt.figure(figsize=(12, 4))
    
    # Plot first layer filters
    filters = model.conv1.weight.data
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(filters[i, 0], cmap='gray')
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    
    plt.suptitle('First Layer CNN Filters')
    plt.tight_layout()
    plt.savefig('cnn_filters.png', dpi=150, bbox_inches='tight')
    print("CNN filters saved as 'cnn_filters.png'")

class SimpleRNN(nn.Module):
    """Simple RNN for sequence data"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
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

def rnn_sequence_demo():
    """Demonstrate RNN for sequence classification"""
    print("\n=== RNN Sequence Classification Demo ===")
    
    # Create synthetic sequence data
    num_samples = 1000
    seq_length = 20
    input_size = 10
    num_classes = 3
    
    # Generate random sequences
    X = torch.randn(num_samples, seq_length, input_size)
    # Create labels based on sequence characteristics (sum of last few elements)
    y = (torch.sum(X[:, -5:, :], dim=(1, 2)) > 0).long()
    
    # Split data
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleRNN(input_size=input_size, hidden_size=64, num_layers=2, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"RNN Architecture:\n{model}")
    print(f"Sequence shape: {X_train.shape}")
    print(f"Sequence length: {seq_length}, Features: {input_size}")
    
    # Training
    model.train()
    num_epochs = 30
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

def transfer_learning_demo():
    """Demonstrate transfer learning concept"""
    print("\n=== Transfer Learning Demo ===")
    
    try:
        import torchvision.models as models
        
        # Load pre-trained ResNet (this would normally be for real images)
        print("Loading pre-trained ResNet-18...")
        resnet = models.resnet18(pretrained=True)
        
        # Freeze all parameters
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Replace final layer for our number of classes
        num_classes = 5
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
        print(f"Modified ResNet for {num_classes} classes")
        print(f"Only final layer will be trained (transfer learning)")
        print(f"Final layer: {resnet.fc}")
        
        # Show which parameters will be updated
        params_to_update = []
        for name, param in resnet.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print(f"Parameter to update: {name}")
        
        print(f"Total parameters to update: {len(params_to_update)}")
        
    except ImportError:
        print("torchvision not available for transfer learning demo")

def optimization_techniques():
    """Demonstrate various optimization techniques"""
    print("\n=== Optimization Techniques ===")
    
    # Create a simple function to optimize
    def rosenbrock(x, y):
        """Rosenbrock function - classic optimization test function"""
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # Different optimizers
    optimizers_to_test = ['SGD', 'Adam', 'RMSprop']
    
    plt.figure(figsize=(15, 5))
    
    for idx, opt_name in enumerate(optimizers_to_test):
        # Initialize parameters
        x = torch.tensor([-1.5], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        
        # Choose optimizer
        if opt_name == 'SGD':
            optimizer = optim.SGD([x, y], lr=0.001)
        elif opt_name == 'Adam':
            optimizer = optim.Adam([x, y], lr=0.01)
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop([x, y], lr=0.01)
        
        # Track optimization path
        x_history, y_history, loss_history = [], [], []
        
        # Optimization loop
        for i in range(1000):
            optimizer.zero_grad()
            loss = rosenbrock(x, y)
            loss.backward()
            optimizer.step()
            
            x_history.append(x.item())
            y_history.append(y.item())
            loss_history.append(loss.item())
        
        # Plot results
        plt.subplot(1, 3, idx + 1)
        plt.plot(loss_history)
        plt.title(f'{opt_name} Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        
        print(f"{opt_name}: Final loss = {loss_history[-1]:.6f}")
        print(f"  Final x = {x_history[-1]:.4f}, y = {y_history[-1]:.4f}")
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
    print("Optimization comparison saved as 'optimization_comparison.png'")

def model_saving_loading():
    """Demonstrate model saving and loading"""
    print("\n=== Model Saving and Loading ===")
    
    # Create a simple model
    model = SimpleNN(10, 64, 3)
    
    # Save model state dict
    torch.save(model.state_dict(), 'simple_model.pth')
    print("Model saved as 'simple_model.pth'")
    
    # Load model
    loaded_model = SimpleNN(10, 64, 3)
    loaded_model.load_state_dict(torch.load('simple_model.pth'))
    loaded_model.eval()
    print("Model loaded successfully")
    
    # Test that models are identical
    test_input = torch.randn(1, 10)
    
    model.eval()
    with torch.no_grad():
        output1 = model(test_input)
        output2 = loaded_model(test_input)
        
    print(f"Original model output: {output1}")
    print(f"Loaded model output: {output2}")
    print(f"Models identical: {torch.allclose(output1, output2)}")
    
    # Save entire model (less flexible but simpler)
    torch.save(model, 'complete_model.pth')
    # Load with weights_only=False for complete model (PyTorch 2.6+)
    loaded_complete = torch.load('complete_model.pth', weights_only=False)
    print("Complete model saved and loaded")

def main():
    """Run all PyTorch examples"""
    print("PyTorch Deep Learning Examples")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    pytorch_fundamentals()
    autograd_demo()
    neural_network_classification()
    cnn_demo()
    rnn_sequence_demo()
    transfer_learning_demo()
    optimization_techniques()
    model_saving_loading()
    
    print("\n" + "=" * 50)
    print("PyTorch tutorial complete!")
    print("Generated files:")
    print("- pytorch_classification_results.png")
    print("- cnn_filters.png") 
    print("- optimization_comparison.png")
    print("- simple_model.pth")
    print("- complete_model.pth")
    print("\nNext steps:")
    print("- Explore the PyTorch documentation")
    print("- Try the Jupyter notebook for interactive learning")
    print("- Experiment with real datasets (MNIST, CIFAR-10)")
    print("- Build your own neural network architectures")

if __name__ == "__main__":
    main()