import numpy as np
import torch.nn.utils
import argparse
from esn_alternative import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_lorenz, get_lorenz_attractor, plot_lorenz_attractor_with_error

# Try running with the following line:
# python3 lorenz.py --test_trials=10 --use_test --rho 1.0 --leaky 0.1 --regul 0.05 --n_hid 512 --inp_scaling 0.8

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
#
parser.add_argument('--bias_scaling', type=float, default=None,
                    help='ESN bias scaling')
parser.add_argument('--avoid_rescal_effective', action="store_false")
parser.add_argument('--solver', type=str, default=None,
                    help='Ridge ScikitLearn solver')



args = parser.parse_args()
print(args)
namefile = 'lorenz_log_ESN'

if args.lag > 1:
    stepahead = '_lag' + str(args.lag)
    namefile += stepahead

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 3 # number of input features
# n_out = 1
washout = 200
lag = args.lag

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=0)

NRMSE = np.zeros(args.test_trials)
for guess in range(args.test_trials):
    model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho, n_layers=2,
                                input_scaling=args.inp_scaling,
                                connectivity_recurrent=args.n_hid,
                                connectivity_input=args.n_hid, 
                                leaky=args.leaky,
                                feedback_size=n_inp
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
        predictions = classifier.predict(activations)
        target = target[washout:]
        plot_lorenz_attractor_with_error(predictions, target, title)
        # print(f"---------- Predictions: {predictions}\n\n")
        # print(f"---------- Target: {target}\n\n\n\n\n\n")
        # calculate nrmse
        mse = np.mean(np.square(predictions - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse

    # columns = ['x', 'y', 'z']
    dataset = train_dataset.unsqueeze(0).reshape(1, -1, 3).to(device) # reshape element to torch.Size([1, rows=len(train_dataset), columns=3])
    target = train_target.reshape(-1, 3).numpy() # reshape element to torch.Size([rows=len(train_target), columns=3])
    # print(f"train dataset: {dataset.shape}")
    # print(f"train target: {target.shape}")
    activations = model(dataset, target)[0].cpu().numpy() # train the deep reservoir to get the activations (states of last iteration combined)
    # print(f"activations: {activations.shape}")
    activations = activations.reshape(-1, args.n_hid) # reshape the activations to torch.Size([rows=len(train_dataset), columns=args.n_hid=256]) (why reshape to 256???)
    activations = activations[washout:] # remove first washout elements (why????)
    # activations = activations.reshape(-1, args.n_hid)
    # print(f"activations: {activations.shape}")
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations) # scale the activations
    # print("target size:", (target.shape))
    # print("activations", (activations.shape))
    target = target[washout:] # remove first washout elements
    if args.solver is None:
        classifier = Ridge(alpha=args.regul, max_iter=1000).fit(activations, target)
    elif args.solver == 'svd':
        classifier = Ridge(alpha=args.regul, solver='svd').fit(activations, target)
    else:
        classifier = Ridge(alpha=args.regul, solver=args.solver).fit(activations, target)
    valid_nmse = test_esn(valid_dataset, valid_target, classifier, scaler, title="Lorenz Attractor Validation Plot") # get nmse of the validation dataset
    test_nmse = test_esn(test_dataset, test_target, classifier, scaler, title="Lorenz Attractor Test Plot") if args.use_test else 0.0 # get nmse of the test dataset
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