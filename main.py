import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esn_alternative import DeepReservoir
from utils import get_lorenz_attractor, get_lorenz96, get_rossler_attractor, compute_nrmse, plot_train_test_prediction_and_target, plot_prediction_distribution, plot_error, plot_prediction_3d, plot_variable_correlations, plot_prediction_2d, compute_dimwise_weights, compute_nrmse_matrix, compute_ks_distances
import numpy as np
import argparse
from sklearn import preprocessing
import torch


# Try running with the following line:
# python3 main.py --system={system} --config_file=config.json
# where {system} = "lorenz" or "lorenz96" or "rossler"


import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None, help='Path to JSON config file')
parser.add_argument('--system', type=str, default="lorenz", help='Dynamic system to simulate: "lorenz", "lorenz96" or "rossler"')


args = parser.parse_args()
SYSTEM = args.system

# If JSON file is provided, override args
if args.config_file is not None:
    with open(SYSTEM + "/" + args.config_file, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

print(config)

namefile = f'{SYSTEM}_log_ESN'

if config["lag"] > 1:
    stepahead = '_lag' + str(config["lag"])
    namefile += stepahead

main_folder = f'{SYSTEM}/results'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

n_inp = config["input_size"] # number of input features
n_out = config["output_size"] # number of output features
washout = config["washout"]
tot_dims = config["total_dimensions"]
predicted_dims = []
for i in range(config["n_modules"]):
    predicted_dims += config["reservoirs"][i]["output_dimensions"]
input_dims = []
for i in range(config["n_modules"]):
    input_dims += config["reservoirs"][i]["input_dimensions"]

input_dims = sorted(list(set(input_dims)))
predicted_dims = sorted(list(set(predicted_dims)))

if SYSTEM == "lorenz":
    (train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=washout, bigger_dataset=config["bigger_dataset"])
elif SYSTEM == "lorenz96":
    (train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz96(N=tot_dims, washout=washout)
elif SYSTEM == "rossler":
    (train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_rossler_attractor(washout=washout)

if config["rescale_input"]:
    scaler = preprocessing.MinMaxScaler().fit(train_dataset)
    train_dataset = torch.tensor(scaler.transform(train_dataset), dtype=torch.float32)
    train_target = torch.tensor(scaler.transform(train_target), dtype=torch.float32)
    valid_dataset = torch.tensor(scaler.transform(valid_dataset), dtype=torch.float32)
    valid_target = torch.tensor(scaler.transform(valid_target), dtype=torch.float32)

NRMSE = np.zeros(config["test_trials"])
for guess in range(config["test_trials"]):
    model = DeepReservoir(config).to(device)

    train_dataset = train_dataset.unsqueeze(0).reshape(1, -1, tot_dims).to(device) # reshape element to torch.Size([1, rows=len(train_dataset), columns=n_inp])
    train_target = train_target.numpy() # reshape element to torch.Size([rows=len(train_target), columns=n_inp])

    # plot_variable_correlations(train_dataset, labels=[f"x{i}" for i in range(tot_dims)])


    model.fit(
        train_dataset, train_target, washout
    ) # train the model's Wout weights feeding it the training dataset


    if "mean_mode" in config and config["mean_mode"]:
        mean_predictions = {j: None for j in predicted_dims}
        train_predictions = [{j: None for j in predicted_dims} for _ in range(model.n_modules)]
        for m in range(model.n_modules):
            activations = model.reservoirs[m].activations
            if config["use_h2"]:
                h2_activations = np.power(activations, 2)
                activations = np.concatenate([activations, h2_activations], axis=1)
            module_predictions = model.reservoirs[m].classifier.predict(
                model.reservoirs[m].scaler.transform(activations)
            )
            k = 0
            if len(model.reservoirs[m].net.output_dimensions) > 1:
                for j in model.reservoirs[m].net.output_dimensions:
                    train_predictions[m][j] = module_predictions[:, k]
                    k += 1
            else:
                j = model.reservoirs[m].net.output_dimensions[0]
                train_predictions[m][j] = module_predictions

        if "weighted_mean" in config and config["weighted_mean"]:
            weights = np.zeros(model.n_modules)
            nrmses_train = np.zeros(model.n_modules)
            for m in range(model.n_modules):
                nrmses_train[m] = compute_nrmse(np.stack([train_predictions[m][j] for j in predicted_dims], axis=1), train_target[washout:])
                print(f"nrmse for module {m}: {nrmses_train[m]}")
            for m in range(model.n_modules):
                weights[m] = ((1/nrmses_train[m]))/np.sum((1/nrmses_train))
                print(f"\n\nweight for module {m}: {weights[m]}\n\n")

        for j in predicted_dims:
            mean_predictions[j] = np.mean([train_predictions[m][j] for m in range(model.n_modules)], axis=0)
        train_predictions = [mean_predictions[j] for j in predicted_dims]
        train_predictions = np.stack(train_predictions, axis=1)

    
    else:
        train_predictions = {j: None for j in predicted_dims}
        for m in range(model.n_modules):
            activations = model.reservoirs[m].activations
            if config["use_h2"]:
                h2_activations = np.power(activations, 2)
                activations = np.concatenate([activations, h2_activations], axis=1)
            module_predictions = model.reservoirs[m].classifier.predict(
                model.reservoirs[m].scaler.transform(activations)
            )
            k = 0
            if len(model.reservoirs[m].net.output_dimensions) > 1:
                for j in model.reservoirs[m].net.output_dimensions:
                    train_predictions[j] = module_predictions[:, k]
                    k += 1
            else:
                j = model.reservoirs[m].net.output_dimensions[0]
                train_predictions[j] = module_predictions
        train_predictions = [train_predictions[j] for j in predicted_dims]
        train_predictions = np.stack(train_predictions, axis=1)

    train_target = train_target[washout:]
    train_dataset = train_dataset[0][washout:] # remove the washout from the dataset

    test_dataset = valid_dataset.unsqueeze(0).reshape(1, -1, tot_dims).to(device)
    test_target = valid_target.numpy()
    n = test_target.shape[0]
    test_target = torch.tensor(test_dataset[0:n], dtype=torch.float32).reshape(-1, tot_dims) # reshape element to torch.Size([rows=len(train_target), columns=3])

    if config["use_self_loop"]:
        if "mean_mode" in config and config["mean_mode"]:
            if "weighted_mean" in config and config["weighted_mean"]:
                test_predictions = np.array(model.mean_predict(n, weights=weights, Y=test_target)).reshape(-1, tot_dims)
            else:
                test_predictions = np.array(model.mean_predict(n, Y=test_target)).reshape(-1, tot_dims)
        else:
            test_predictions = np.array(model.predict(n, Y=test_target)).reshape(-1, tot_dims) # get the model's prediction for n iterations
    else:
        test_predictions = np.array(model.teacher_forcing_predict(n, u_init=train_dataset[-1, :].reshape(1, 1, -1), Y=test_target)).reshape(-1, tot_dims)
    
    test_target = test_target.numpy()
    NRMSE = [compute_nrmse(test_predictions, test_target)] # compute nrmse for each prediction

    config["error"] = float(np.mean(NRMSE))

    # train_predictions = train_predictions[:, predicted_dims]
    test_predictions = test_predictions[:, predicted_dims]
    train_target = train_target[:, predicted_dims]
    test_target = test_target[:, predicted_dims]

    print(f"test predictions shape: {test_predictions.shape}")
    print(f"test target shape: {test_target.shape}")

    labels = [f"x{i}" for i in predicted_dims] if SYSTEM == "lorenz96" else ["x", "y", "z"]

    ks_distances = compute_ks_distances(test_predictions, test_target)
    print(f"\n\n\nKolmogorov-Smirnov distances per dimensione: {ks_distances}\n\n\n")
    config["ks_distance"] = {labels[i]: ks_distances[i] for i in range(len(predicted_dims))}

    # Overwrite the config file with updated content
    with open(SYSTEM + "/" + args.config_file, 'w') as f:
        json.dump(config, f, indent=4)

    plot_train_test_prediction_and_target(train_predictions, train_target, test_predictions, test_target, inp_dim=len(predicted_dims), labels=labels) if config["show_plot"] else None
    plot_error(test_predictions, test_target, n_dim=len(predicted_dims), labels=labels) if config["show_plot"] else None
    plot_prediction_distribution(train_predictions, train_target, "Train", labels=labels) if config["show_plot"] else None
    plot_prediction_distribution(test_predictions, test_target, "Test", labels=labels) if config["show_plot"] else None
    plot_prediction_3d(test_predictions, test_target, title=f"{SYSTEM.capitalize()} Attractor", labels=labels) if config["show_plot"] and len(predicted_dims) == 3 else None
    plot_prediction_2d(test_predictions, test_target, title=f"{SYSTEM.capitalize()} Attractor", labels=[labels[i] for i in predicted_dims]) if config["show_plot"] and len(predicted_dims) == 2 else None






mean = np.mean(NRMSE)
std = np.std(NRMSE)
lastprint = ' ##################################################################### \n'
lastprint += 'Mean NRMSE ' + str(mean) + ',    std ' + str(std) + '\n'
lastprint += ' ##################################################################### \n'
print(lastprint)
f = open(f'{main_folder}/{namefile}.txt', 'a')
f.write(lastprint)
f.close()

# # store new experiment to csv
# try:
#     result_dataset = pd.read_csv("./results/lorenz_results.csv")
# except FileNotFoundError:
#     result_dataset = pd.DataFrame(columns=["n_hid", "inp_scaling", "rho", "leaky", "regul", "lag", "bias_scaling", "solver", "washout", "n_layers", "NRMSE_mean, NRMSE_std"])

# new_row = {
#     "n_hid": config.n_hid,
#     "inp_scaling": config.inp_scaling,
#     "rho": config.rho,
#     "leaky": config.leaky,
#     "regul": config.regul,
#     "lag": lag,
#     "bias_scaling": config.bias_scaling,
#     "solver": config.solver,
#     "washout": washout,
#     "n_layers": n_modules,
#     "NRMSE_mean": mean,
#     "NRMSE_std": std
# }

# result_dataset = pd.concat([result_dataset, pd.DataFrame([new_row])], ignore_index=True)
# result_dataset.to_csv("./results/lorenz_results.csv", index=False)