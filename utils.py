import torch
import matplotlib.pyplot as plt
from pathlib import Path

WEIGHTS_PATH = Path("Model_Weights")
GAN_WEIGHTS_PATH = WEIGHTS_PATH / "GAN"
TRPO_WEIGHTS_PATH = WEIGHTS_PATH / "TRPO"

EXPERT_DATA_PATH = Path("Expert_Trajectories")
EXPERT_DATA_STATES_PATH = EXPERT_DATA_PATH / "states.pt"
EXPERT_DATA_ACTIONS_PATH = EXPERT_DATA_PATH / "actions.pt"
EXPERT_DATA_REWARDS_PATH = EXPERT_DATA_PATH / "rewards.pt"
EXPERT_DATA_NEXT_STATES_PATH = EXPERT_DATA_PATH / "next_states.pt"

def normalize(A, mean, std):
    # mean = A.mean(dim=0)
    # std = A.std(dim=0) + 1e-8
    A = (A - mean) / std
    return A

def to_tensor(A):
    return torch.tensor(A, dtype=torch.float32).unsqueeze(0)

def print_tensor_statistics(tensor):
    """
    Prints statistics (min, max, mean, std) for each column in a 2D tensor.
    
    Parameters:
        tensor (numpy.ndarray or similar): A 2D array-like object.
    """
    # Print header for clarity
    print(f"{'Index':<5} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 45)
    
    # Loop and print the formatted values
    for i in range(tensor.shape[1]):
        print(f"{i:<5} {tensor[:, i].min():>10.4f} {tensor[:, i].max():>10.4f} {tensor[:, i].mean():>10.4f} {tensor[:, i].std():>10.4f}")

def plot_histogram(data_points, bins=10, title="Histogram of Data Points", xlabel="Data Points"):
    """
    Plots a histogram to visualize the distribution of data points.

    Parameters:
        data_points (list or array): The data points to plot.
        bins (int): Number of bins for the histogram (default: 10).
        title (str): Title of the plot (default: "Histogram of Data Points").
        xlabel (str): Label for the x-axis (default: "Data Points").
    """
    plt.figure(figsize=(10, 5))
    plt.hist(data_points, bins=bins, edgecolor="black", alpha=0.7)

    # Beautify the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Show the plot
    plt.show()