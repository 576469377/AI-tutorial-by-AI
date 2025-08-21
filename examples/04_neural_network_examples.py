"""
Neural Network Examples
Implementation and visualization of neural networks from scratch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_circles
import warnings
warnings.filterwarnings('ignore')

class SimpleNeuralNetwork:
    """A simple neural network implementation from scratch"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """Initialize the neural network with random weights"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases randomly
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # Store training history
        self.loss_history = []
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        
        # Calculate gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-15) + (1 - y) * np.log(1 - output + 1e-15))
            self.loss_history.append(loss)
            
            # Backward propagation
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

def visualize_activation_functions():
    """Visualize different activation functions"""
    print("=== Activation Functions ===")
    
    x = np.linspace(-5, 5, 100)
    
    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    # Plot activation functions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid(x), 'b-', linewidth=2, label='Sigmoid')
    plt.title('Sigmoid Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x, tanh(x), 'r-', linewidth=2, label='Tanh')
    plt.title('Tanh Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(x, relu(x), 'g-', linewidth=2, label='ReLU')
    plt.title('ReLU Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu(x), 'm-', linewidth=2, label='Leaky ReLU')
    plt.title('Leaky ReLU Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Activation functions visualized and saved as 'activation_functions.png'")

def neural_network_from_scratch():
    """Demonstrate neural network implementation from scratch"""
    print("\n=== Neural Network from Scratch ===")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Reshape y to be a column vector
    y = y.reshape(-1, 1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)
    nn.train(X_train_scaled, y_train, epochs=1000)
    
    # Make predictions
    train_predictions = nn.predict(X_train_scaled)
    test_predictions = nn.predict(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(nn.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot decision boundary
    plt.subplot(1, 3, 2)
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train.ravel(), 
                         cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot test results
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test.ravel(), 
                         cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter)
    plt.title(f'Test Data\nAccuracy: {test_accuracy:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('neural_network_results.png', dpi=100, bbox_inches='tight')
    plt.show()

def visualize_network_architecture():
    """Visualize neural network architecture"""
    print("\n=== Network Architecture Visualization ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simple network visualization
    ax1 = axes[0]
    
    # Layer positions
    input_y = [3, 2, 1]
    hidden_y = [3.5, 2.5, 1.5, 0.5]
    output_y = [2]
    
    input_x = [0] * len(input_y)
    hidden_x = [2] * len(hidden_y)
    output_x = [4] * len(output_y)
    
    # Draw nodes
    ax1.scatter(input_x, input_y, s=300, c='lightblue', edgecolors='black', label='Input Layer')
    ax1.scatter(hidden_x, hidden_y, s=300, c='lightgreen', edgecolors='black', label='Hidden Layer')
    ax1.scatter(output_x, output_y, s=300, c='lightcoral', edgecolors='black', label='Output Layer')
    
    # Draw connections
    for i, iy in enumerate(input_y):
        for j, hy in enumerate(hidden_y):
            ax1.plot([0, 2], [iy, hy], 'k-', alpha=0.3, linewidth=1)
    
    for i, hy in enumerate(hidden_y):
        for j, oy in enumerate(output_y):
            ax1.plot([2, 4], [hy, oy], 'k-', alpha=0.3, linewidth=1)
    
    # Add labels
    for i, y in enumerate(input_y):
        ax1.text(-0.3, y, f'X{i+1}', fontsize=12, ha='center', va='center')
    
    for i, y in enumerate(hidden_y):
        ax1.text(2, y, f'H{i+1}', fontsize=10, ha='center', va='center')
    
    ax1.text(4.3, output_y[0], 'Y', fontsize=12, ha='center', va='center')
    
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(0, 4)
    ax1.set_title('Simple Neural Network\n(3-4-1 Architecture)')
    ax1.legend(loc='upper right')
    ax1.axis('off')
    
    # Deep network visualization
    ax2 = axes[1]
    
    # More complex architecture
    layers = [4, 6, 6, 3, 1]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output']
    colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightgreen', 'lightcoral']
    
    for layer_idx, (size, name, color) in enumerate(zip(layers, layer_names, colors)):
        x = layer_idx * 1.5
        y_positions = np.linspace(0, 3, size)
        
        ax2.scatter([x] * size, y_positions, s=200, c=color, edgecolors='black')
        
        # Draw connections to next layer
        if layer_idx < len(layers) - 1:
            next_size = layers[layer_idx + 1]
            next_y_positions = np.linspace(0, 3, next_size)
            
            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax2.plot([x, x + 1.5], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)
        
        # Add layer labels
        ax2.text(x, -0.5, name, fontsize=10, ha='center', va='center', rotation=45)
    
    ax2.set_xlim(-0.5, len(layers) * 1.5)
    ax2.set_ylim(-1, 4)
    ax2.set_title('Deep Neural Network\n(4-6-6-3-1 Architecture)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Network architectures visualized and saved as 'network_architecture.png'")

def gradient_descent_visualization():
    """Visualize gradient descent optimization"""
    print("\n=== Gradient Descent Visualization ===")
    
    # Create a simple 2D loss landscape
    def loss_function(w1, w2):
        return (w1 - 2)**2 + (w2 - 1)**2 + 0.1 * np.sin(5 * w1) * np.cos(5 * w2)
    
    # Create mesh for contour plot
    w1 = np.linspace(-1, 5, 100)
    w2 = np.linspace(-2, 4, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = loss_function(W1, W2)
    
    # Simulate gradient descent
    def gradient(w1, w2):
        dw1 = 2 * (w1 - 2) + 0.5 * np.cos(5 * w1) * np.cos(5 * w2)
        dw2 = 2 * (w2 - 1) - 0.5 * np.sin(5 * w1) * np.sin(5 * w2)
        return dw1, dw2
    
    # Different learning rates
    learning_rates = [0.1, 0.01, 0.3]
    colors = ['red', 'blue', 'green']
    
    plt.figure(figsize=(12, 4))
    
    for i, (lr, color) in enumerate(zip(learning_rates, colors)):
        plt.subplot(1, 3, i + 1)
        
        # Plot loss landscape
        plt.contour(W1, W2, Z, levels=20, alpha=0.6)
        plt.contourf(W1, W2, Z, levels=20, alpha=0.3, cmap='viridis')
        
        # Gradient descent path
        w1_current, w2_current = 0.0, 3.0  # Starting point
        path_w1, path_w2 = [w1_current], [w2_current]
        
        for step in range(50):
            dw1, dw2 = gradient(w1_current, w2_current)
            w1_current -= lr * dw1
            w2_current -= lr * dw2
            path_w1.append(w1_current)
            path_w2.append(w2_current)
            
            if np.sqrt(dw1**2 + dw2**2) < 1e-6:
                break
        
        # Plot path
        plt.plot(path_w1, path_w2, color=color, linewidth=2, marker='o', 
                markersize=3, label=f'LR={lr}')
        plt.plot(path_w1[0], path_w2[0], 'ro', markersize=8, label='Start')
        plt.plot(path_w1[-1], path_w2[-1], 'g*', markersize=12, label='End')
        
        plt.xlabel('Weight 1')
        plt.ylabel('Weight 2')
        plt.title(f'Learning Rate = {lr}')
        plt.legend()
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('gradient_descent.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Gradient descent visualization saved as 'gradient_descent.png'")

def main():
    """Run all neural network examples"""
    print("Neural Network Examples")
    print("=" * 50)
    
    # Visualize activation functions
    visualize_activation_functions()
    
    # Neural network from scratch
    neural_network_from_scratch()
    
    # Visualize network architecture
    visualize_network_architecture()
    
    # Gradient descent visualization
    gradient_descent_visualization()
    
    print("\n" + "=" * 50)
    print("Neural network examples complete!")
    print("\nKey concepts learned:")
    print("1. Activation functions add non-linearity to neural networks")
    print("2. Forward propagation computes predictions")
    print("3. Backward propagation computes gradients for learning")
    print("4. Learning rate affects how fast and stable the training is")
    print("5. Network architecture (layers, neurons) affects model capacity")

if __name__ == "__main__":
    main()