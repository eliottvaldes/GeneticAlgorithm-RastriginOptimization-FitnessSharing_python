# import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Rastrigin function
def rastrigin_function(individual: np.ndarray) -> float:
    x, y = individual
    term_1 = np.power(x, 2) - 10 * np.cos(10 * np.pi * x)
    term_2 = np.power(y, 2) - 10 * np.cos(2 * np.pi * y)
    return 20 + (term_1 + term_2)

# Plot Rastrigin function 3D and contour side by side
def plot_rastrigin_3d_and_contour():
    # Create meshgrid
    x = np.linspace(-5.12, 5.12, 80)
    y = np.linspace(-5.12, 5.12, 80)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values
    Z = np.array([rastrigin_function(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    
    # Create a figure with two subplots (one for 3D, one for contour)
    fig = plt.figure(figsize=(12, 6))
    
    # Left subplot: 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, rstride=1, cstride=1, alpha=0.9, edgecolor='none')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title('Rastrigin Function (3D)')
    ax1.view_init(45, 45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Right subplot: Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, 20, cmap=cm.viridis)
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_title('Rastrigin Function (Contour)')
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
    
    # Show both plots at the same time
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_rastrigin_3d_and_contour()
