import numpy as np
import subprocess
from scipy.stats import qmc
import time

# Define parameter ranges
n_samples = 1000  # Number of experiments to run
max_parallel = 4

# Latin Hypercube Sampler
sampler = qmc.LatinHypercube(d=9)
sample = sampler.random(n=n_samples)

# Scale to parameter ranges
leaky_range = [0.9, 1.0]
scaling_range = [0.0, 1.0]
units_range = [200, 500]

def scale(param, bounds):
    return param * (bounds[1] - bounds[0]) + bounds[0]

# Discretize units to closest step of 50
def round_units(u):
    return int(round(u / 50) * 50)

# Open results file
out_file = open("results.txt", "w")

# Prepare all parameter sets
configs = []
for row in sample:
    l1, l2, l3 = [scale(row[i], leaky_range) for i in range(3)]
    s1, s2, s3 = [scale(row[i], scaling_range) for i in range(3, 6)]
    u1, u2, u3 = [round_units(scale(row[i], units_range)) for i in range(6, 9)]

    config = {
        "leaky": (l1, l2, l3),
        "scaling": (s1, s2, s3),
        "units": (u1, u2, u3),
    }
    configs.append(config)

# Run in parallel batches
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_experiment(config):
    l1, l2, l3 = config["leaky"]
    s1, s2, s3 = config["scaling"]
    u1, u2, u3 = [512, 512, 1024]

    cmd = [
        "python3", "lorenz.py",
        "--test_trials=1",
        "--use_test",
        "--rho", "0.9",
        "--leaky", "0.715",
        "--regul", "0.000002",
        "--n_hid", "750",
        "--inp_scaling", "0.11",
        "--washout", "200",
        "--use_self_loop",
        "--n_modules", "3",
        "--rescale_input",
        "--mode", "entangled",
        f"--leaky1={l1}", f"--leaky2={l2}", f"--leaky3={l3}",
        f"--inp_scaling1={s1}", f"--inp_scaling2={s2}", f"--inp_scaling3={s3}",
        f"--units1={u1}", f"--units2={u2}", f"--units3={u3}"
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        output = e.output

    return {
        "config": config,
        "output": output
    }

with ThreadPoolExecutor(max_workers=max_parallel) as executor:
    futures = [executor.submit(run_experiment, cfg) for cfg in configs]

    for future in as_completed(futures):
        result = future.result()
        c = result["config"]
        out_file.write(f"\n--- Config ---\nleaky: {c['leaky']}\ninp_scaling: {c['scaling']}\nunits: {c['units']}\n")
        out_file.write(result["output"])
        out_file.write("\n\n")

out_file.close()
print("✅ All LHS experiments completed. See results.txt")