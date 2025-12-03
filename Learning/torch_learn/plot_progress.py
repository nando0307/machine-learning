import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    """
    Adjusted for PyCharm: Uses plt.pause() instead of clear_output
    """
    # Create predictions
    predicted_norm = model(distances_norm).detach().numpy()
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()

    # Clear the current figure without closing the window
    plt.clf()

    # Plot data
    plt.plot(x_plot, y_plot, color='orange', marker='o', linestyle='none', label='Actual Data')

    # Sort for clean line plotting
    sorted_indices = x_plot.argsort(axis=0).flatten()
    plt.plot(x_plot[sorted_indices], predicted_norm[sorted_indices], color='green', label='Model Predictions')

    # dynamic title
    current_loss = loss.item() if torch.is_tensor(loss) else loss
    plt.title(f'Epoch: {epoch + 1} | Loss: {current_loss:.4f}')
    plt.xlabel('Normalized Distance')
    plt.ylabel('Normalized Time')
    plt.legend()
    plt.grid(True)

    # CRITICAL FOR PYCHARM ANIMATION:
    # pause(0.01) updates the window and waits 0.01s.
    # It replaces plt.show() and clear_output()
    plt.pause(0.01)