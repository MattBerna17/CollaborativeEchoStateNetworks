import numpy as np
import torch.nn.utils
import argparse
from esn_alternative import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_lorenz_attractor, plot_lorenz_attractor_with_error, save_matrix_to_file, plot_prediction_and_target, compute_nrmse, plot_error, plot_train_test_prediction_and_target, plot_variable_correlations, plot_reservoir_state_2d, plot_reservoirs_states
import pandas as pd
import matplotlib.pyplot as plt
import torch


# Try running with the following line:
# python3 lorenz.py --test_trials=1 --use_test --rho 0.9 --leaky 0.9 --regul 0.000001 --n_hid 256 --inp_scaling 0.014 --washout 200 --n_layers 1 --use_self_loop --show_plot



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



(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=washout, bigger_dataset=config["bigger_dataset"])

print(train_dataset.shape)

if config["skip_z"]:
    if config["mode"] != "entangled_with_z":
        train_dataset = train_dataset[:, :-1].reshape(-1, 2)
        valid_dataset = valid_dataset[:, :-1].reshape(-1, 2)
        test_dataset = test_dataset[:, :-1].reshape(-1, 2)
    train_target = train_target[:, :-1].reshape(-1, 2)
    valid_target = valid_target[:, :-1].reshape(-1, 2)
    test_target = test_target[:, :-1].reshape(-1, 2)

scaler = preprocessing.MinMaxScaler().fit(train_dataset)
if config["rescale_input"]:
    train_dataset = torch.tensor(scaler.transform(train_dataset), dtype=torch.float32)
    train_target = torch.tensor(scaler.transform(train_target), dtype=torch.float32)
    valid_dataset = torch.tensor(scaler.transform(valid_dataset), dtype=torch.float32)
    valid_target = torch.tensor(scaler.transform(valid_target), dtype=torch.float32)

NRMSE = np.zeros(config["test_trials"])
for guess in range(config["test_trials"]):
    model = DeepReservoir(config).to(device)

    # no_grad means that the operations inside the block will not be added to the computation graph
    # since we never use torch.backward() we don't need to compute the gradient
    @torch.no_grad()
    def test_esn(dataset, target, classifier, scaler, title):
        # reshape the dataset and the target
        dataset = dataset.unsqueeze(0).reshape(1, -1, 3).to(device)
        target = target.reshape(-1, 3).numpy()
        activations = model(dataset)[0].cpu().numpy() # calculate activations and reshape + remove washout
        activations = activations.reshape(-1, config["n_hid"])
        activations = activations[washout:]
        activations = scaler.transform(activations)
        # save_matrix_to_file(activations, title + "_activations") # to save the activations from the model
        predictions = classifier.predict(activations)
        target = target[washout:]
        plot_error(torch.from_numpy(predictions), target) if config["show_plot"] else None
        plot_prediction_and_target(predictions, target) if config["show_plot"] else None
        # calculate nrmse
        mse = np.mean(np.square(predictions - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse


    train_dataset = train_dataset.unsqueeze(0).reshape(1, -1, n_inp).to(device) # reshape element to torch.Size([1, rows=len(train_dataset), columns=n_inp])
    train_target = train_target.reshape(-1, n_out).numpy() # reshape element to torch.Size([rows=len(train_target), columns=n_inp])
    


    model.fit(
        train_dataset, train_target, washout
    ) # train the model's Wout weights feeding it the training dataset
    
    
    training_activations = [model.reservoirs[i].activations for i in range(model.n_modules)] # take the reservoirs' activations during training



    if config["n_modules"] > 1:
        train_predictions = [None for _ in range(n_out)]
        if model.mode == "entangled_with_z":
            train_predictions = [None, None, None]
            train_predictions[1] = model.reservoirs[0].classifier.predict(
                model.reservoirs[0].scaler.transform(model.reservoirs[0].activations)
            )
            train_predictions[0] = model.reservoirs[1].classifier.predict(
                model.reservoirs[1].scaler.transform(model.reservoirs[1].activations)
            )
            train_predictions[2] = model.reservoirs[2].classifier.predict(
                model.reservoirs[2].scaler.transform(model.reservoirs[2].activations)
            )
        for m in range(model.n_modules):
            if model.mode == "entangled":
                train_predictions[(m + 1)%n_out] = model.reservoirs[m].classifier.predict(
                    model.reservoirs[m].scaler.transform(model.reservoirs[m].activations)
                )
        train_predictions = np.stack(train_predictions, axis=1) # stack predictions to torch.Size([rows=len(train_dataset), columns=n_out])
    else:
        train_predictions = model.reservoirs[0].classifier.predict(
            model.reservoirs[0].scaler.transform(model.reservoirs[0].activations)
        )

    train_target = train_target[washout:]
    train_dataset = train_dataset[0][washout:] # remove the washout from the dataset

    # plot_prediction_and_target(train_predictions, train_target[:, 0:2], inp_dim=2) if config["show_plot"] else None # plot the prediction


    test_dataset = valid_dataset.unsqueeze(0).reshape(1, -1, n_inp).to(device)
    test_target = valid_target.reshape(-1, n_out).numpy()
    n = test_target.shape[0]
    test_target = torch.tensor(test_dataset[0:n], dtype=torch.float32).reshape(-1, n_out) # reshape element to torch.Size([rows=len(train_target), columns=3])
    print(test_target[0])

    if config["n_modules"] > 1:
        test_predictions = model.predict(n, Y=test_target).numpy() # get the model's prediction for n iterations
    else:
        test_predictions = np.array(model.predict(n, Y=test_target)) # get the model's prediction for n iterations
    testing_activations = [model.reservoirs[i].activations[-n:, :] for i in range(model.n_modules)] # take the reservoirs' activations during testing
    # for m in range(model.n_modules):
    #     plot_reservoir_state_2d(training_activations[m], testing_activations[m], reservoir_index=m)
    
    
    test_target = test_target.numpy()
    # train_target = train_target[:, 0:2]
    NRMSE = [compute_nrmse(test_predictions, test_target)] # compute nrmse for each prediction
    # plot_train_test_prediction_and_target(train_predictions, train_target, test_predictions, test_target, inp_dim=n_out, train_activations_list=training_activations, test_activations_list=testing_activations) if config["show_plot"] else None

    plot_train_test_prediction_and_target(train_predictions, train_target, test_predictions, test_target, inp_dim=3) if config["show_plot"] else None
    # plot_prediction_and_target(test_predictions, test_target, inp_dim=2) if config["show_plot"] else None # plot the prediction

    # valid_nmse = test_esn(valid_dataset, valid_target, classifier, scaler, title="validation") # get nmse of the validation dataset
    # test_nmse = test_esn(test_dataset, test_target, classifier, scaler, title="test") if config.use_test else 0.0 # get nmse of the test dataset
    # NRMSE[guess] = test_nmse
    # f = open(f'{main_folder}/{namefile}.txt', 'a')
    # ar = ''
    # for k, v in vars(config).items():
    #     ar += f'{str(k)}: {str(v)}, '
    # ar += f'valid: {str(round(valid_nmse, 5))}, test: {str(round(test_nmse, 5))}'
    # f.write(ar + '\n')
    # f.write('**************\n\n\n')
    # f.close()






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