import pandas as pd
import matplotlib.pyplot as plt

def generate_rossler_dataset(a=0.2, b=0.2, c=5.7, dt=0.01, steps=10000, x0=0., y0=1., z0=0.):
    """
    Generate the Rössler attractor data using the Runge-Kutta 4th order method.
    """
    def rossler_deriv(x, y, z):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return dx, dy, dz

    x, y, z = x0, y0, z0
    data = []

    for i in range(steps):
        t = i * dt
        dx1, dy1, dz1 = rossler_deriv(x, y, z)
        dx2, dy2, dz2 = rossler_deriv(x + 0.5 * dt * dx1, y + 0.5 * dt * dy1, z + 0.5 * dt * dz1)
        dx3, dy3, dz3 = rossler_deriv(x + 0.5 * dt * dx2, y + 0.5 * dt * dy2, z + 0.5 * dt * dz2)
        dx4, dy4, dz4 = rossler_deriv(x + dt * dx3, y + dt * dy3, z + dt * dz3)

        x += (dt / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
        y += (dt / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
        z += (dt / 6.0) * (dz1 + 2 * dz2 + 2 * dz3 + dz4)

        data.append([t, x, y, z])

    return pd.DataFrame(data, columns=['t', 'x', 'y', 'z'])


# Generate and save the dataset
df = generate_rossler_dataset()
df.to_csv("rossler_dataset.csv", index=False)
print("✅ Dataset saved as 'rossler_dataset.csv'.")

# Optional: plot to visualize the attractor
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')
ax.plot(df['x'], df['y'], df['z'], lw=0.5)
ax.set_title("Rössler Attractor")
plt.show()