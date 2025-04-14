#!/bin/bash

output_file="scaling_results.txt"
> "$output_file"

scalings=(0.05 0.1 0.15 0.2 0.25 0.3)

counter=0

for s1 in "${scalings[@]}"; do
  for s2 in "${scalings[@]}"; do
    for s3 in "${scalings[@]}"; do
      ((counter++))
      echo "[$counter] Running with scaling1=$s1, scaling2=$s2, scaling3=$s3"

      output=$(python3 lorenz.py \
        --test_trials=1 \
        --use_test \
        --rho 0.9 \
        --leaky 0.715 \
        --regul 0.000002 \
        --n_hid 750 \
        --inp_scaling 0.11 \
        --washout 200 \
        --use_self_loop \
        --n_modules 3 \
        --mode entangled \
        --leaky1=1.0 --leaky2=1.0 --leaky3=1.0 \
        --inp_scaling1=$s1 --inp_scaling2=$s2 --inp_scaling3=$s3 \
        --units1=250 --units2=300 --units3=1024 2>&1)

      echo "[$counter] scaling1=$s1, scaling2=$s2, scaling3=$s3" >> "$output_file"
      echo "$output" >> "$output_file"
      echo "-----------------------------------------------" >> "$output_file"
    done
  done
done

echo "✅ All combinations finished. See $output_file"