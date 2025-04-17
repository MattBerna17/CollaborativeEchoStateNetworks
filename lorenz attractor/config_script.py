import json
import subprocess
import itertools
import os
import random

# Optimized ranges
leaky_r1 = [0.8, 0.9, 1.0]
scaling_r1 = [0.05, 0.1, 0.15]
regul_r1 = [1e-4, 1e-3]
units_r1 = [300, 400, 500]
rho_r1 = [0.9]

leaky_r2 = [0.7, 0.8, 0.9]
scaling_r2 = [0.05, 0.1, 0.2]
regul_r2 = [1e-4, 1e-3, 1e-2]
units_r2 = [300, 500, 600]
rho_r2 = [0.9]

output_file = "config_output.txt"
if os.path.exists(output_file):
    os.remove(output_file)

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
}

r1_configs = list(itertools.product(leaky_r1, scaling_r1, regul_r1, units_r1, rho_r1))
r2_configs = list(itertools.product(leaky_r2, scaling_r2, regul_r2, units_r2, rho_r2))

# Sample 100 random combinations of (R1, R2)
all_combos = list(itertools.product(r1_configs, r2_configs))
random.seed(42)
sampled_combos = random.sample(all_combos, 100)

for count, ((l1, s1, r1, u1, rho1), (l2, s2, r2, u2, rho2)) in enumerate(sampled_combos, 1):
    config = base_config.copy()
    config["reservoirs"] = [
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": l1,
            "inp_scaling": s1,
            "regul": r1,
            "units": u1,
            "rho": rho1,
            "solver": None
        },
        {
            "input_size": 1,
            "output_size": 1,
            "leaky": l2,
            "inp_scaling": s2,
            "regul": r2,
            "units": u2,
            "rho": rho2,
            "solver": None
        }
    ]

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
    print(f"✅ Run {count}/100 complete")

print(f"\n🎉 Done! 100 configurations saved in {output_file}")