import json
import subprocess
import itertools
import os

# Limited ranges for Reservoirs 0 & 1
leaky_small = [0.6, 0.7]
scaling_small = [0.05, 0.1]
regul_small = [1e-6, 3e-6]
rho_small = [0.9, 1.0]

# Full ranges for Reservoir 2
leaky_full = [0.6, 0.7, 0.8, 1.0]
scaling_full = [0.05, 0.1, 0.2, 0.3]
regul_full = [1e-6, 3e-6, 1e-5, 1e-2]
units_full = [250, 300, 500, 750]
rho_full = [0.9, 1.0, 1.1]

output_file = "config_output.txt"
if os.path.exists(output_file):
    os.remove(output_file)

base_config = {
    "lag": 1,
    "bigger_dataset": False,
    "test_trials": 1,
    "n_modules": 3,
    "use_self_loop": True,
    "washout": 200,
    "mode": "entangled",
    "rescale_input": False,
    "show_plot": False,
    "input_size": 3,
    "output_size": 3,
    "concat": False
}

r0_units, r1_units = 250, 300

r0_configs = list(itertools.product(leaky_small, scaling_small, regul_small, rho_small))
r1_configs = list(itertools.product(leaky_small, scaling_small, regul_small, rho_small))
r2_configs = list(itertools.product(leaky_full, scaling_full, regul_full, units_full, rho_full))

count = 0
for i, (l0, s0, r0, rho0) in enumerate(r0_configs):
    for j, (l1, s1, r1, rho1) in enumerate(r1_configs):
        for k, (l2, s2, r2, u2, rho2) in enumerate(r2_configs):
            count += 1
            config = base_config.copy()
            config["reservoirs"] = [
                {
                    "input_size": 1,
                    "output_size": 1,
                    "leaky": l0,
                    "inp_scaling": s0,
                    "regul": r0,
                    "units": r0_units,
                    "rho": rho0,
                    "solver": None
                },
                {
                    "input_size": 1,
                    "output_size": 1,
                    "leaky": l1,
                    "inp_scaling": s1,
                    "regul": r1,
                    "units": r1_units,
                    "rho": rho1,
                    "solver": None
                },
                {
                    "input_size": 2,
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
            print(f"✅ Run {count} complete")

print(f"🎉 Done! All results are in {output_file}")