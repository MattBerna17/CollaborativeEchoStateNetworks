import numpy as np
import torch.nn.utils
import argparse
from esn_alternative import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_mackey_glass

# Try running with the following line:
# python forecast_mackeyglass.py --test_trials=10 --use_test --rho 1. --inp_scaling 1 --leaky 0.9 --regul 1e-6 --lag 1 --n_hid 100 --solver svd


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN leakage')
parser.add_argument('--regul', type=float, default=0.0,
                    help='Ridge regularisation parameter')
parser.add_argument('--lag', type=int, default=1)
parser.add_argument('--use_test', action="store_true")
parser.add_argument('--show_result', type=bool, default=False)
parser.add_argument('--test_trials', type=int, default=1,
                    help='number of trials to compute mean and std on test')
#
parser.add_argument('--bias_scaling', type=float, default=None,
                    help='ESN bias scaling')
parser.add_argument('--avoid_rescal_effective', action="store_false")
parser.add_argument('--solver', type=str, default=None,
                    help='Ridge ScikitLearn solver')



args = parser.parse_args()
print(args)
namefile = 'mackey_log_ESN'

if args.lag > 1:
    stepahead = '_lag' + str(args.lag)
    namefile += stepahead

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 1
n_out = 1
washout = 200
lag = args.lag

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_mackey_glass(args.lag)

NRMSE = np.zeros(args.test_trials)
for guess in range(args.test_trials):


    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                                input_scaling=args.inp_scaling,
                                connectivity_recurrent=args.n_hid,
                                connectivity_input=args.n_hid, 
                                leaky=args.leaky,
                                ).to(device)


    @torch.no_grad()
    def test_esn(dataset, target, classifier, scaler):
        dataset = dataset.reshape(1, -1, 1).to(device)
        target = target.reshape(-1, 1).numpy()
        activations = model(dataset)[0].cpu().numpy()
        activations = activations[:, washout:]
        activations = activations.reshape(-1, args.n_hid)
        activations = scaler.transform(activations)
        predictions = classifier.predict(activations)
        mse = np.mean(np.square(predictions - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse

    dataset = train_dataset.reshape(1, -1, 1).to(device)
    target = train_target.reshape(-1, 1).numpy()
    
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    

    if args.solver is None:
        classifier = Ridge(alpha=args.regul, max_iter=1000).fit(activations, target)
    elif args.solver == 'svd':
        classifier = Ridge(alpha=args.regul, solver='svd').fit(activations, target)
    else:
        classifier = Ridge(alpha=args.regul, solver=args.solver).fit(activations, target)
    valid_nmse = test_esn(valid_dataset, valid_target, classifier, scaler)
    test_nmse = test_esn(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0
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