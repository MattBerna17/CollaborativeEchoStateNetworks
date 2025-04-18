import json
import subprocess
import itertools
import os
import random

# Full regul sweep range
all_regul_values = [i * 10**exp for exp in range(-6, 0) for i in range(1, 10)]  # 45 values total

# Sample 20 distinct values for each reservoir
random.seed(42)
regul_r1 = random.sample(all_regul_values, 20)
regul_r2 = random.sample(all_regul_values, 20)

# Base config to modify
base_config = {
    "lag": 1,
    "bigger_dataset": False,
    "test_trials": 1,
    "n_modules": 2,
    "use_self_loop": True,
    "washout": 200,
    "mode": "entangled_with_z",
    "rescale_input": False,
    "show_plot": False,
    "input_size": 3,
    "output_size": 3,
    "concat": False,
    "skip_z": False,
    "reservoirs": [
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.1,
            "inp_scaling": 0.05,
            "regul": 7e-1,  # Placeholder
            "units": 250,
            "rho": 1.1,
            "solver": None
        },
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": 0.1,
            "inp_scaling": 0.181,
            "regul": 4e-3,  # Placeholder
            "units": 500,
            "rho": 1.1,
            "solver": None
        }
    ]
}

output_file = "config_output_20x20.txt"
if os.path.exists(output_file):
    os.remove(output_file)

count = 0
for r1_regul, r2_regul in itertools.product(regul_r1, regul_r2):
    count += 1
    config = json.loads(json.dumps(base_config))  # Deep copy
    config["reservoirs"][0]["regul"] = r1_regul
    config["reservoirs"][1]["regul"] = r2_regul

    filename = f"config_tmp_{count}.json"
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)

    try:
        result = subprocess.run(
            ["python3", "lorenz.py", "--config_file", filename],
            capture_output=True,
            text=True,
            timeout=300
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        output = f"Timeout for config {filename}"

    with open(output_file, "a") as f:
        f.write(f"\n--- Config File: {filename} ---\n")
        f.write(json.dumps(config, indent=2) + "\n")
        f.write(output + "\n")

    os.remove(filename)
    print(f"✅ Run {count}/400 complete")

print(f"\n🎉 Done! All {count} configurations saved in {output_file}")