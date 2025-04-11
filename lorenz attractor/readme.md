# ESNs for predicting Lorenz Attractor
## Run
The following command is used to execute the lorenz attractor predictor using Echo State Networks with only 1 reservoir (layer in the following) to predict the next state of the attractor:

```bash
python3 lorenz.py --test_trials=1 --use_test --rho 0.9 --leaky 0.9 --regul 0.000003 --n_hid 256 --inp_scaling 0.014 --washout 200 --n_layers 1 --use_self_loop --show_plot
```

## Versions
### Version 1.0 - Single reservoir
In the first version of this project, we tried to use only 1 reservoir, that took as inputs all the dimensions at time $t$, i.e. $u(t) = [x(t), y(t), z(t)]$ to predict the values for $u(t+1) = [x(t+1), y(t+1), z(t+1)]$ using a 256-dimensional representation of the input applying non-linear transformations. During training, it is given, at each timestep, the ground truth values, but during testing it takes as input the past prediction, and starts this generative loop.

This version works perfectly and is capable of reproducing the climate of the attractor, which means that, even though it does not predict exactly the ground truth value during test, it keeps behaving like the original lorenz attractor.


### Version 2.0 - Deep reservoir
In the second version, the single reservoir of the original code is split into 3 reservoir (layers), each one taking as input the $i$-th dimension of the lorenz attractor and predicting the $i+1$-th (i.e. the first reservoir takes as input $x(t)$ and predicts $y(t+1)$, and so on).

During training, each reservoir is fed with the ground truth values, and during testing it takes as input the prediction of the previous (in a circular way) reservoir.

This version does not work correctly: during training we can see that the last reservoir (input $z(t)$ output $x(t+1)$) cannot reproduce the correct values of $x(t+1)$, and that causes the quick explosion of activations during testing.