import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from esn_alternative import DeepReservoir
from utils import get_lorenz_attractor, compute_nrmse, plot_train_test_prediction_and_target, plot_prediction_distribution, plot_error, plot_prediction_3d
import numpy as np
import argparse
from sklearn import preprocessing
import torch


# Try running with the following line:
# python3 lorenz.py --config_file=config.json



import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None, help='Path to JSON config file')

# (Include your existing argparse arguments here...)

args = parser.parse_args()

# If JSON file is provided, override args
if args.config_file is not None:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

print(config)

namefile = 'lorenz_log_ESN'

if config["lag"] > 1:
    stepahead = '_lag' + str(config["lag"])
    namefile += stepahead

main_folder = 'results'

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

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=washout, bigger_dataset=config["bigger_dataset"])

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


    model.fit(
        train_dataset, train_target, washout
    ) # train the model's Wout weights feeding it the training dataset

    train_predictions = {j: None for j in predicted_dims}
    for m in range(model.n_modules):
        module_predictions = model.reservoirs[m].classifier.predict(
            model.reservoirs[m].scaler.transform(model.reservoirs[m].activations)
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

    test_predictions = np.array(model.predict(n, Y=test_target)).reshape(-1, tot_dims) # get the model's prediction for n iterations
    
    test_target = test_target.numpy()
    NRMSE = [compute_nrmse(test_predictions, test_target)] # compute nrmse for each prediction

    train_predictions = train_predictions[:, predicted_dims]
    test_predictions = test_predictions[:, predicted_dims]
    train_target = train_target[:, predicted_dims]
    test_target = test_target[:, predicted_dims]


    plot_train_test_prediction_and_target(train_predictions, train_target, test_predictions, test_target, inp_dim=len(predicted_dims)) if config["show_plot"] else None
    plot_error(test_predictions, test_target, n_dim=len(predicted_dims)) if config["show_plot"] else None
    plot_prediction_distribution(train_predictions, train_target, "Train") if config["show_plot"] else None
    plot_prediction_distribution(test_predictions, test_target, "Test") if config["show_plot"] else None
    plot_prediction_3d(test_predictions, test_target, title="Lorenz Attractor") if config["show_plot"] and len(predicted_dims) == 3 else None






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