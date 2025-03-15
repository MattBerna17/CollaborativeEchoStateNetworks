import numpy as np

def process_matrix(filename):
    min_value = float("inf")
    max_value = float("-inf")
    
    # Process the file line by line to find min and max values
    with open(filename, "r") as file:
        data = []
        for line in file:
            values = np.array([float(num) for num in line.strip().split(",") if num])  # Avoid empty strings
            min_value = min(min_value, values.min())
            max_value = max(max_value, values.max())
            data.append(values)

    # Convert to NumPy array
    matrix = np.vstack(data)
    
    # Compute condition number using singular values
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    condition_number = S.max() / S.min()

    print(f"Min Value: {min_value}")
    print(f"Max Value: {max_value}")
    print(f"Condition Number: {condition_number}")

# Example usage
filename = "./data/03_15__15_20/03_15_15_20_25___train_activations.txt"  # Replace with actual file path
process_matrix(filename)
