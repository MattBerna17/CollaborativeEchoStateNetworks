import numpy as np
import pandas as pd

def lorenz_attractor(sigma=10, beta=8/3, rho=28, dt=0.01, num_steps=10000):
    # Initial conditions
    x, y, z = 0.0, 5.0, -8.0
    data = []

    for i in range(num_steps):
        # Lorenz equations
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        # Update state using Euler method
        x += dx * dt
        y += dy * dt
        z += dz * dt

        t = i * dt  # Time step
        data.append([i, t, x, y, z])

    # Save to CSV
    df = pd.DataFrame(data, columns=["", "t", "x", "y", "z"])
    df.to_csv("lorenz_attractor_10000.csv", index=False)

    print("Generated Lorenz attractor dataset with 10,000 rows.")

# Run the function
lorenz_attractor()