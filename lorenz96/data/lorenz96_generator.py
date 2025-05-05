import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from scipy.integrate import odeint
import pandas as pd
from utils import get_fixed_length_windows

def lorenz96_generator(N, F, num_batch=128, lag=1, washout=200, window_size=0):
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        d = np.zeros(N)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    dt = 0.01
    t = np.arange(0.0, 20 + (lag * dt) + (washout * dt), dt)
    dataset = []
    for i in range(num_batch):
        x0 = np.random.rand(N) + F - 0.5
        x = odeint(L96, x0, t)
        dataset.append(x)

    dataset = np.stack(dataset, axis=0)  # [batch, time, dim]
    dataset_tensor = torch.from_numpy(dataset).float()

    # Save to CSV
    num_total = dataset.shape[0]
    with open("lorenz96.csv", "w") as f:
        f.write("batch,time," + ",".join([f"x{i}" for i in range(N)]) + "\n")
        for b in range(num_total):
            for ti, row in enumerate(dataset[b]):
                f.write(f"{b},{t[ti]}," + ",".join(map(str, row)) + "\n")

    if window_size > 0:
        windows, targets = [], []
        for i in range(dataset_tensor.shape[0]):
            w, t = get_fixed_length_windows(dataset_tensor[i], window_size, prediction_lag=lag)
            windows.append(w)
            targets.append(t)
        return torch.utils.data.TensorDataset(torch.cat(windows, dim=0), torch.cat(targets, dim=0))
    else:
        return dataset_tensor


if __name__ == "__main__":
    N = 4         # Number of dimensions (typical for Lorenz96)
    F = 8.0       # Forcing term (commonly used for chaotic behavior)
    num_batch = 128  # Number of trajectories to generate
    lag = 1
    washout = 200
    window_size = 0  # Set to > 0 if you want windowed output for training

    print(f"Generating Lorenz96 data with N={N}, F={F}, lag={lag}, batches={num_batch}")
    dataset = lorenz96_generator(N, F, num_batch=num_batch, lag=lag, washout=washout, window_size=window_size)
    print("✅ Dataset saved to 'lorenz96.csv'")