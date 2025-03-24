#!/bin/bash

# Output file
output_file="model_selection_results.txt"
> "$output_file"  # Clear file before starting

# Parameter values
rho_values=(0.8 0.9 1.0)
leaky_values=(0.7 0.8 0.9 1.0)
regul_values=(0.000001 0.000002 0.000003)
n_hid_values=(100 200 256 300)
inp_scaling_values=(0.009 0.010 0.011 0.012 0.013 0.014)

# Run all combinations
for rho in "${rho_values[@]}"; do
    for leaky in "${leaky_values[@]}"; do
        for regul in "${regul_values[@]}"; do
            for n_hid in "${n_hid_values[@]}"; do
                for inp_scaling in "${inp_scaling_values[@]}"; do

                    # Command to run
                    command="python3 ./../../lorenz.py --test_trials=1 --use_test --rho $rho --leaky $leaky --regul $regul --n_hid $n_hid --inp_scaling $inp_scaling --washout 200 --n_layers 1 --use_self_loop"
                    echo "Running: $command"

                    # Run the command and capture the output
                    output=$($command 2>&1)

                    # Save results
                    echo -e "--------------------------------------" >> "$output_file"
                    echo -e "rho: $rho\nleaky: $leaky\nregul: $regul\nn_hid: $n_hid\ninp_scaling: $inp_scaling" >> "$output_file"
                    echo -e "Output:\n$output\n" >> "$output_file"

                done
		echo -e "\n" >> "$output_file"
            done
	    echo -e "\n" >> "$output_file"
        done
	echo -e "\n" >> "$output_file"
    done
    echo -e "\n" >> "$output_file"
done

echo "All experiments completed. Results saved in $output_file"
