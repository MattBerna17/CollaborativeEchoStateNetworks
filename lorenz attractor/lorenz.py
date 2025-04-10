import numpy as np
import torch.nn.utils
import argparse
from esn_alternative import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_lorenz_attractor, plot_lorenz_attractor_with_error, save_matrix_to_file, plot_prediction_and_target, compute_nrmse, plot_error, plot_train_test_prediction_and_target, plot_variable_correlations
import pandas as pd
import matplotlib.pyplot as plt
import torch


# Try running with the following line:
# python3 lorenz.py --test_trials=1 --use_test --rho 0.9 --leaky 0.9 --regul 0.000001 --n_hid 256 --inp_scaling 0.014 --washout 200 --n_layers 1 --use_self_loop --show_plot



parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net') # 512 might be a better value but is significantly slower
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling') # 0.5 seems to be a good value
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius') # doesn't change the result, at least for the lorenz attractor
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN leakage') # 0.1 seems to be the best value
parser.add_argument('--regul', type=float, default=0.0,
                    help='Ridge regularisation parameter') # around 0.1 and 0.05 seems to be the best value
parser.add_argument('--lag', type=int, default=1)
parser.add_argument('--use_test', action="store_true")
parser.add_argument('--show_result', type=bool, default=False)
parser.add_argument('--test_trials', type=int, default=1,
                    help='number of trials to compute mean and std on test')
parser.add_argument('--feedback_size', type=int, default=0,
                    help='Number of connections the feedback matrix is going to have')
parser.add_argument('--neighbour_feedback_size', type=int, default=0,
                    help='Number of connections the neighbours\' feedback matrix is going to have')
parser.add_argument('--bias_scaling', type=float, default=None,
                    help='ESN bias scaling')
parser.add_argument('--avoid_rescal_effective', action="store_false")
parser.add_argument('--solver', type=str, default=None,
                    help='Ridge ScikitLearn solver')
parser.add_argument('--washout', type=int, default=250,
                    help='Number of washout iterations')
parser.add_argument('--show_plot', action="store_true",
                    help='Whether to show the plot of the prediction compared to the actual function')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of layers in the deep reservoir')
parser.add_argument('--neighbour_scaling', type=float, default=1.,
                    help='ESN neighbour feedback scaling')
parser.add_argument('--bigger_dataset', action="store_true",
                    help='If specified, uses the bigger dataset (lorenz_attractor_10000.csv) instead of the standard one (lorenz.csv)')
parser.add_argument('--use_self_loop', action="store_true",
                    help='If specified, uses the feedback loop: the output at timestep t is given as input at timestep t+1 (i.e. o(t) = u(t+1))')



args = parser.parse_args()
print(args)
namefile = 'lorenz_log_ESN'

if args.lag > 1:
    stepahead = '_lag' + str(args.lag)
    namefile += stepahead

main_folder = 'results'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 3 # number of input features
# n_out = 1
washout = args.washout
lag = args.lag
show_plot = args.show_plot
feedback_size = args.feedback_size
n_layers = args.n_layers
neighbour_feedback_size = args.neighbour_feedback_size
neighbour_scaling = args.neighbour_scaling if not(neighbour_feedback_size == 0) else 0
bigger_dataset = args.bigger_dataset
use_self_loop = args.use_self_loop

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=washout, bigger_dataset=bigger_dataset)

NRMSE = np.zeros(args.test_trials)
for guess in range(args.test_trials):
    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho, n_layers=n_layers,
                                input_scaling=args.inp_scaling,
                                connectivity_recurrent=args.n_hid,
                                connectivity_input=args.n_hid, 
                                leaky=args.leaky
                                ).to(device)

    # no_grad means that the operations inside the block will not be added to the computation graph
    # since we never use torch.backward() we don't need to compute the gradient
    @torch.no_grad()
    def test_esn(dataset, target, classifier, scaler, title):
        # reshape the dataset and the target
        dataset = dataset.unsqueeze(0).reshape(1, -1, 3).to(device)
        target = target.reshape(-1, 3).numpy()
        activations = model(dataset)[0].cpu().numpy() # calculate activations and reshape + remove washout
        activations = activations.reshape(-1, args.n_hid)
        activations = activations[washout:]
        activations = scaler.transform(activations)
        # save_matrix_to_file(activations, title + "_activations") # to save the activations from the model
        predictions = classifier.predict(activations)
        target = target[washout:]
        plot_error(torch.from_numpy(predictions), target) if show_plot else None
        plot_prediction_and_target(predictions, target) if show_plot else None
        # calculate nrmse
        mse = np.mean(np.square(predictions - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse


    # columns = ['x', 'y', 'z']
    dataset = train_dataset.unsqueeze(0).reshape(1, -1, 3).to(device) # reshape element to torch.Size([1, rows=len(train_dataset), columns=3])
    target = train_target.reshape(-1, 3).numpy() # reshape element to torch.Size([rows=len(train_target), columns=3])
    
    
    scalers, classifiers = model.train(dataset, target, args.washout, args.solver, args.regul) # train the model's Wout weights feeding it the training dataset
    if n_layers > 1:
        train_predictions = [None for _ in range(n_inp)]
        for l in range(model.n_layers):
            train_predictions[(l+1)%model.n_layers] = classifiers[l].predict(scalers[l].transform(model.reservoir[l].activations))
        train_predictions = np.stack(train_predictions, axis=1)
    else:
        train_predictions = classifiers[0].predict(scalers[0].transform(model.reservoir[0].activations))

    train_target = target[washout:]
    dataset = dataset[0][washout:] # remove the washout from the dataset

    # plot_variable_correlations(dataset)

    print(f"[INPUT] {dataset}")
    print(f"##########################")
    print(f"[TRAINING PREDICTION] {train_predictions}") # print the first 5 predictions
    print(f"\n\n\n##########################\n\n\n")
    print(f"[TRAINING GROUND TRUTH] {train_target}\n\n\n\n") # print the first 5 targets
    plot_prediction_and_target(train_predictions, train_target) if show_plot else None # plot the prediction
    # print(f"[TRAINING PREDICTION] {train_predictions[-5:]}") # print the first 5 predictions
    # print(f"[TRAINING GROUND TRUTH] {train_target[-5:]}") # print the first 5 targets
    # print("\n\n\n")
    # print(f"[COEFFICIENTS] {[np.linalg.norm(classifiers[l].coef_) for l in range(n_layers)]}") # print the coefficients of the classifier
    # print(f"[INTERCEPTS] {[classifiers[l].intercept_ for l in range(n_layers)]}") # print the intercept of the classifier


    print(f"###########################################################")
    print(f"######################## DEBUG ############################")
    print(f"###########################################################")
    # print(f"Prediction of reservoir 0 on 2.2534: {model.reservoir[0].classifier.predict(model.reservoir[0].forward(torch.tensor([2.2534]).reshape(1, 1, 1)))}\n\n")


    dataset = valid_dataset.unsqueeze(0).reshape(1, -1, 3).to(device)
    target = valid_target.reshape(-1, 3).numpy()
    # assert train_dataset[-1] == dataset[0][0]

    if use_self_loop:
        n = target.shape[0]
        # n = 5
        # target = target[:n]
        target = dataset[0:n].reshape(-1, 3).numpy() # reshape element to torch.Size([rows=len(train_target), columns=3])
        # print(f"[GROUND TRUTH] {dataset[0:n]}")
        print(f"[TRAIN PRED] {train_predictions[-1]}")
        predictions = model.predict(n, y_init=train_target[-1], Y=target) # get the model's prediction for n iterations
        # print(predictions)
        # predictions = predictions[washout:] # remove the washout
        # target = target[washout:] # remove the washout
        NRMSE = [compute_nrmse(predictions, target)] # compute nrmse for each prediction
        predictions = torch.stack(predictions)
        test_predictions = predictions
        test_target = target
        plot_train_test_prediction_and_target(train_predictions, train_target, test_predictions, test_target) if show_plot else None
        # plot_error(predictions, target) if show_plot else None # plot the error
        # plot_prediction(predictions) if show_plot else None
        plot_prediction_and_target(predictions, target) if show_plot else None # plot the prediction
    
    
    
    
    else:
        valid_nmse = test_esn(valid_dataset, valid_target, classifier, scaler, title="validation") # get nmse of the validation dataset
        test_nmse = test_esn(test_dataset, test_target, classifier, scaler, title="test") if args.use_test else 0.0 # get nmse of the test dataset
        NRMSE[guess] = test_nmse
        f = open(f'{main_folder}/{namefile}.txt', 'a')
        ar = ''
        for k, v in vars(args).items():
            ar += f'{str(k)}: {str(v)}, '
        ar += f'valid: {str(round(valid_nmse, 5))}, test: {str(round(test_nmse, 5))}'
        f.write(ar + '\n')
        f.write('**************\n\n\n')
        f.close()


        if args.show_result:
            print(ar)






mean = np.mean(NRMSE)
std = np.std(NRMSE)
lastprint = ' ##################################################################### \n'
lastprint += 'Mean NRMSE ' + str(mean) + ',    std ' + str(std) + '\n'
lastprint += ' ##################################################################### \n'
print(lastprint)
f = open(f'{main_folder}/{namefile}.txt', 'a')
f.write(lastprint)
f.close()

# store new experiment to csv
try:
    result_dataset = pd.read_csv("./results/lorenz_results.csv")
except FileNotFoundError:
    result_dataset = pd.DataFrame(columns=["n_hid", "inp_scaling", "rho", "leaky", "regul", "lag", "bias_scaling", "solver", "washout", "n_layers", "NRMSE_mean, NRMSE_std"])

new_row = {
    "n_hid": args.n_hid,
    "inp_scaling": args.inp_scaling,
    "rho": args.rho,
    "leaky": args.leaky,
    "regul": args.regul,
    "lag": lag,
    "bias_scaling": args.bias_scaling,
    "solver": args.solver,
    "washout": washout,
    "n_layers": n_layers,
    "NRMSE_mean": mean,
    "NRMSE_std": std
}

result_dataset = pd.concat([result_dataset, pd.DataFrame([new_row])], ignore_index=True)
result_dataset.to_csv("./results/lorenz_results.csv", index=False)