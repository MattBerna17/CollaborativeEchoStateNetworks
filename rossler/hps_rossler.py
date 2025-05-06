import json
import subprocess
import os
import datetime

# Create base directory for outputs
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"experiments/run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Base config
base_config = {
    "total_dimensions": 3,
    "lag": 1,
    "bigger_dataset": False,
    "test_trials": 1,
    "n_modules": 2,
    "use_self_loop": True,
    "washout": 200,
    "rescale_input": False,
    "show_plot": True,  # <- disables display
    "input_size": 2,
    "output_size": 2,
    "concat": False,
    "reservoirs": [
        {
            "input_size": 1,
            "output_size": 1,
            "input_dimensions": [0],
            "output_dimensions": [1],
            "leaky": 0.06,
            "inp_scaling": 0.4,
            "regul": 1e-2,
            "units": 324,
            "rho": 1.1,
            "solver": None
        },
        {
            "input_size": 1,
            "output_size": 1,
            "input_dimensions": [1],
            "output_dimensions": [0],
            "leaky": 0.06,
            "inp_scaling": 0.47,
            "regul": 1e-2,
            "units": 365,
            "rho": 1.1,
            "solver": None
        }
    ]
}

# Values to sweep (for example: different leaky rates for module 0)
leaky_values = [0.01 * i for i in range(5, 21)]  # 0.05 to 0.20

# Run experiments
for count, leaky in enumerate(leaky_values, 1):
    config = base_config.copy()
    config["reservoirs"][0]["leaky"] = leaky  # Change module 0's leaky value
    config_name = f"config_leaky{leaky:.3f}.json"
    config_path = os.path.join(output_dir, "configs", config_name)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n🔁 Running with leaky = {leaky:.3f}...")
    try:
        subprocess.run(
            ["python3", "rossler.py", "--config_file", config_path, "--save_folder", os.path.join(output_dir, "plots")],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error with leaky = {leaky:.3f}: {e}")