import json
import subprocess
import os

# Base config template
base_config = {
    "lag": 1,
    "bigger_dataset": False,
    "test_trials": 1,
    "n_modules": 3,
    "use_self_loop": True,
    "washout": 200,
    "mode": "EMP-3",
    "rescale_input": False,
    "show_plot": True,
    "input_size": 3,
    "output_size": 3,
    "concat": False,
    "skip_z": False,
    "reservoirs": [
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.08,
            "inp_scaling": 0.05,
            "regul": 1,
            "units": 400,
            "rho": 1.1,
            "solver": None
        },
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.008,
            "inp_scaling": 0.1,
            "regul": 1,
            "units": 300,
            "rho": 1.1,
            "solver": None
        },
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.026,
            "inp_scaling": 0.16,
            "regul": 1e-3,
            "units": 410,
            "rho": 1.1,
            "solver": None
        }
    ]
  }

# Create output directory if it doesn't exist
os.makedirs("configs", exist_ok=True)

# Define the regularization values to test
# units_values = [0 + i for i in range(100, 800, 10)]
leaky_values = [0.001*i for i in range(5, 21)]

# Loop over regularization values
for count, leaky in enumerate(leaky_values, 1):
    # base_config["reservoirs"][0]["leaky"] = leaky
    base_config["reservoirs"][1]["leaky"] = leaky
    # base_config["reservoirs"][2]["leaky"] = leaky
    tmp_path = "configs/tmp_config.json"

    with open(tmp_path, "w") as f:
        json.dump(base_config, f, indent=2)

    print(f"\n🔁🌀 Running with leaky = {leaky}")
    try:
        subprocess.run(
            ["python3", "rossler.py", "--config_file", tmp_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error for leaky = {leaky}: {e}")