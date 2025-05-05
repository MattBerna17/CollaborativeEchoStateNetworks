import json
import subprocess
import os

# Base config template
base_config = {
    "lag": 25,
    "bigger_dataset": False,
    "test_trials": 1,
    "n_modules": 2,
    "use_self_loop": True,
    "washout": 200,
    "mode": "CDR",
    "rescale_input": False,
    "show_plot": True,
    "input_size": 2,
    "output_size": 2,
    "concat": False,
    "skip_z": True,
    "reservoirs": [
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.055,
            "inp_scaling": 0.05,
            "regul": 0.7,
            "units": 150,
            "rho": 1.1,
            "solver": None
        },
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.055,
            "inp_scaling": 0.181,
            "regul": 4e-3,
            "units": 500,
            "rho": 1.1,
            "solver": None
        }
    ]
  }

# Create output directory if it doesn't exist
os.makedirs("configs", exist_ok=True)

# Define the regularization values to test
units_values = [0 + i for i in range(100, 300, 10)]

# Loop over regularization values
for count, units in enumerate(units_values, 1):
    # base_config["reservoirs"][0]["units"] = units
    base_config["reservoirs"][1]["units"] = units
    tmp_path = "configs/tmp_config.json"

    with open(tmp_path, "w") as f:
        json.dump(base_config, f, indent=2)

    print(f"\n🔁 🧠 Running with units = {units}")
    try:
        subprocess.run(
            ["python3", "lorenz96.py", "--config_file", tmp_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error for units = {units}: {e}")