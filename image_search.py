import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        # Initialize weights with Xavier/Glorot initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros(output_dim)
        
        # Storage for activations and gradients (for visualization)
        self.z1 = None  # Pre-activation of hidden layer
        self.h1 = None  # Post-activation of hidden layer
        self.z2 = None  # Pre-activation of output layer
        self.gradients = {}  # Store gradients for visualization

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

        
    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sx = 1 / (1 + np.exp(-x))
            return sx * (1 - sx)

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h1 = self.activation(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        out = np.tanh(self.z2)  # Using tanh for final activation
        
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]
        
        # Output layer gradients
        d_z2 = (self.forward(X) - y) * (1 - np.tanh(self.z2)**2)  # derivative of tanh
        d_W2 = np.dot(self.h1.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0) / m
        
        # Hidden layer gradients
        d_h1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_h1 * self.activation_derivative(self.z1)
        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0) / m
        
        # TODO: update weights with gradient descent
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        
        # TODO: store gradients for visualization
        self.gradients = {
            'W1': d_W1,
            'W2': d_W2,
            'b1': d_b1,
            'b2': d_b2
        }

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    current_step = (frame + 1) * 10
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.h1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                     c=y.ravel(), cmap='bwr', alpha=0.7)
    
    # Create a grid of points on the plane with fixed bounds
    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    
    # Calculate z coordinates based on learned weights
    w = mlp.W2.flatten()
    b = mlp.b2[0]
    z = (-w[0] * xx - w[1] * yy - b) / w[2]
    
    # Plot the plane with yellow highlighting
    ax_hidden.plot_surface(xx, yy, z, alpha=0.2, color='yellow')  # Light yellow
        
    # Set fixed bounds and labels
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)
    
    # Set ticks every 0.5
    ax_hidden.set_xticks(np.arange(-1, 1.5, 0.5))
    ax_hidden.set_yticks(np.arange(-1, 1.5, 0.5))
    ax_hidden.set_zticks(np.arange(-1, 1.5, 0.5))
    
    # Add grid
    ax_hidden.grid(True, alpha=0.3)
    
    ax_hidden.set_xlabel('h₁')
    ax_hidden.set_ylabel('h₂')
    ax_hidden.set_zlabel('h₃')
    ax_hidden.set_title(f'Hidden Space at Step {current_step}')
    
    # Add decision surface in hidden space
    xx = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 20)
    yy = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 20)
    zz = np.linspace(hidden_features[:, 2].min(), hidden_features[:, 2].max(), 20)
    grid = np.meshgrid(xx, yy, zz)
    
    # Create a grid in input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions and create binary decision boundary
    Z = mlp.forward(grid_points)
    Z = (Z > 0).reshape(xx.shape)  # Binary decision
    
    # Plot decision boundary with solid colors (swapped colors)
    ax_input.contourf(xx, yy, Z, levels=[-1, 0.5, 1], colors=['blue', 'red'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', s=20)
    ax_input.set_title(f'Input Space at Step {current_step}')
    ax_input.set_xlabel('x₁')
    ax_input.set_ylabel('x₂')
    
    # Visualize network architecture with labeled nodes
    # Define node positions more precisely
    node_positions = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y':  (1.0, 0.5)
    }
    
    # Plot nodes as circles with labels
    for name, pos in node_positions.items():
        ax_gradient.add_patch(Circle(pos, 0.05, color='blue', alpha=0.8))
        ax_gradient.text(pos[0]-0.02, pos[1]-0.15, name, fontsize=10)
    
    # Plot edges with gradient-based thickness
    max_thickness = 2.0
    # Input to hidden connections
    input_nodes = ['x1', 'x2']
    hidden_nodes = ['h1', 'h2', 'h3']
    
    for i, in_node in enumerate(input_nodes):
        for j, hid_node in enumerate(hidden_nodes):
            thickness = np.abs(mlp.gradients['W1'][i, j]) / np.abs(mlp.gradients['W1']).max() * max_thickness
            ax_gradient.plot([node_positions[in_node][0], node_positions[hid_node][0]], 
                           [node_positions[in_node][1], node_positions[hid_node][1]], 
                           'purple', linewidth=thickness, alpha=0.6)
    
    # Hidden to output connections
    for i, hid_node in enumerate(hidden_nodes):
        thickness = np.abs(mlp.gradients['W2'][i, 0]) / np.abs(mlp.gradients['W2']).max() * max_thickness
        ax_gradient.plot([node_positions[hid_node][0], node_positions['y'][0]], 
                        [node_positions[hid_node][1], node_positions['y'][1]], 
                        'purple', linewidth=thickness, alpha=0.6)
    
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.set_title(f'Gradients at Step {current_step}')
    ax_gradient.axis('equal')
    
    # Add grid lines
    ax_gradient.grid(True, linestyle='--', alpha=0.3)
    ax_gradient.set_xticks(np.arange(0, 1.1, 0.2))
    ax_gradient.set_yticks(np.arange(0, 1.1, 0.2))
    
    # Make grid lines appear behind other elements
    ax_gradient.set_axisbelow(True)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, 
                                   ax_gradient=ax_gradient, X=X, y=y), 
                       frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)