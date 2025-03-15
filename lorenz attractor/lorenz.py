import numpy as np
import torch.nn.utils
import argparse
from esn_alternative import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_lorenz, get_lorenz_attractor, plot_lorenz_attractor_with_error, save_matrix_to_file
import pandas as pd

# Try running with the following line:
# python3 lorenz.py --test_trials=10 --use_test --rho 1.0 --leaky 0.1 --regul 0.05 --n_hid 512 --inp_scaling 0.8


def save_models_weights(W_in, W, W_out, filenames):
    with open(filenames[0], "w") as f:
        for i in range(W_in.shape[0]):
            for j in range(W_in.shape[1]):
                f.write(str(W_in[i][j].item()) + ',')
            f.write('\n')
    with open(filenames[1], "w") as f:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                f.write(str(W[i][j].item()) + ',')
            f.write('\n')
    with open(filenames[2], "w") as f:
        for i in range(W_out.shape[0]):
            for j in range(W_out.shape[1]):
                f.write(str(W_out[i][j]) + ',')
            f.write('\n')


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




# !TODO: add feedback kernel scaling param



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

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=washout)

NRMSE = np.zeros(args.test_trials)
for guess in range(args.test_trials):
    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho, n_layers=n_layers,
                                input_scaling=args.inp_scaling,
                                connectivity_recurrent=args.n_hid,
                                connectivity_input=args.n_hid, 
                                leaky=args.leaky,
                                feedback_size=feedback_size,
                                neighbour_feedback_size=neighbour_feedback_size
                                ).to(device)

    # no_grad means that the operations inside the block will not be added to the computation graph
    # since we never use torch.backward() we don't need to compute the gradient
    @torch.no_grad()
    def test_esn(dataset, target, classifier, scaler, title):
        # reshape the dataset and the target
        dataset = dataset.unsqueeze(0).reshape(1, -1, 3).to(device)
        target = target.reshape(-1, 3).numpy()
        activations = model(dataset, target)[0].cpu().numpy() # calculate activations and reshape + remove washout
        activations = activations.reshape(-1, args.n_hid)
        activations = activations[washout:]
        activations = scaler.transform(activations)

        save_matrix_to_file(activations, title + "_activations")
        # W_in = model.reservoir[0].net.kernel.transpose(0, 1)
        # W = model.reservoir[0].net.recurrent_kernel # get the weights of the reservoir
        # W_out = classifier.coef_
        # save_models_weights(W_in, W, W_out, ["W_in.txt", "W.txt", "W_out.txt"])
        predictions = classifier.predict(activations)
        target = target[washout:]
        # print(f"Predictions: {predictions[:10]}\n\n")
        # print(f"Target: {target[:10]}\n\n")
        plot_lorenz_attractor_with_error(predictions, target, title) if show_plot else None
        # calculate nrmse
        mse = np.mean(np.square(predictions - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse

    # columns = ['x', 'y', 'z']
    dataset = train_dataset.unsqueeze(0).reshape(1, -1, 3).to(device) # reshape element to torch.Size([1, rows=len(train_dataset), columns=3])
    target = train_target.reshape(-1, 3).numpy() # reshape element to torch.Size([rows=len(train_target), columns=3])
    activations = model(dataset, target)[0].cpu().numpy() # train the deep reservoir to get the activations (states of last iteration combined)
    activations = activations.reshape(-1, args.n_hid) # reshape the activations to torch.Size([rows=len(train_dataset), columns=args.n_hid=256]) (why reshape to 256???)
    activations = activations[washout:] # remove first washout elements (why????)
    # activations = activations.reshape(-1, args.n_hid)
    print(activations.shape)
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations) # scale the activations
    save_matrix_to_file(activations, "train_activations")

    target = target[washout:] # remove first washout elements
    if args.solver is None:
        classifier = Ridge(alpha=args.regul, max_iter=1000).fit(activations, target)
    elif args.solver == 'svd':
        classifier = Ridge(alpha=args.regul, solver='svd').fit(activations, target)
    else:
        classifier = Ridge(alpha=args.regul, solver=args.solver).fit(activations, target)
    
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
try:
    result_dataset = pd.read_csv("./results/lorenz_result.csv")
except FileNotFoundError:
    result_dataset = pd.DataFrame(columns=["n_hid", "inp_scaling", "rho", "leaky", "regul", "lag", "bias_scaling", "solver", "washout", "feedback_size", "n_layers", "neighbour_size", "NRMSE_mean, NRMSE_std"])

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
    "feedback_size": feedback_size,
    "n_layers": n_layers,
    "neighbour_size": neighbour_feedback_size,
    "NRMSE_mean": mean,
    "NRMSE_std": std
}

result_dataset = pd.concat([result_dataset, pd.DataFrame([new_row])], ignore_index=True)
result_dataset.to_csv("./results/lorenz_result.csv", index=False)