import torch.nn.utils
import argparse
import numpy as np
from scipy.integrate import odeint
import torch # this serves to numerically solve ODEs
from esn_alternative import DeepReservoir
import matplotlib.pyplot as plt
import pandas as pd
from utils import get_fixed_length_windows
from sklearn import preprocessing
from sklearn.linear_model import Ridge


def read_lorenz(lag=1, washout=200, window_size=0):
    dataset = pd.read_csv("lorenz.csv").drop(columns=["Unnamed: 0"]).values
    dataset = torch.tensor(dataset).float()
    if window_size > 0:
        assert washout == 0
        dataset, targets = get_fixed_length_windows(dataset, window_size, prediction_lag=lag)
        # dataset is input, targets is output

        end_train = int(dataset.shape[0] / 2)
        end_val = end_train + int(dataset.shape[0] / 4)
        end_test = dataset.shape[0]

        train_dataset = dataset[:end_train]
        train_target = targets[:end_train]

        val_dataset = dataset[end_train:end_val]
        val_target = targets[end_train:end_val]

        test_dataset = dataset[end_val:end_test]
        test_target = targets[end_val:end_test]
    else:
        end_train = int(dataset.shape[0] / 2)
        end_val = end_train + int(dataset.shape[0] / 4)
        end_test = dataset.shape[0]

        train_dataset = dataset[:end_train-lag]
        train_target = dataset[washout+lag:end_train] # washout ... ?

        # print(f"Train dataset: {(train_dataset)}")
        # print(f"Train target length: {(train_target.shape)}")

        val_dataset = dataset[end_train:end_val-lag]
        val_target = dataset[end_train+washout+lag:end_val]

        test_dataset = dataset[end_val:end_test-lag]
        test_target = dataset[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)
    


if __name__ == "__main__":
    # parse terminal args
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
    parser.add_argument('--bias_scaling', type=float, default=None,
                        help='ESN bias scaling')
    parser.add_argument('--avoid_rescal_effective', action="store_false")
    parser.add_argument('--solver', type=str, default=None,
                        help='Ridge ScikitLearn solver')
    args = parser.parse_args()
    namefile = 'lorenz_log_ESN'

    if args.lag > 1:
        stepahead = '_lag' + str(args.lag)
        namefile += stepahead

    main_folder = 'result'

    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    print("Using device", device)
    n_inp = 1
    n_out = 1
    washout = 200
    lag = args.lag

    # get lorenz dataset
    (train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = read_lorenz(lag, washout)
    # print(train_dataset.shape)
    # print(train_target.shape)

    NRMSE = np.zeros(args.test_trials)
    for guess in range(args.test_trials):
        model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                                    input_scaling=args.inp_scaling,
                                    connectivity_recurrent=args.n_hid,
                                    connectivity_input=args.n_hid,
                                    leaky=args.leaky,
                                    ).to(device)
        
        preds, gts = [pd.Series(), pd.Series()] # to store predictions and ground truth
        
        @torch.no_grad()
        def test_esn(dataset, target, classifier, scaler):
            dataset = dataset.reshape(1, -1, 1).to(device)
            target = target.reshape(-1, 1).numpy()
            activations = model(dataset)[0].cpu().numpy()
            activations = activations[:, washout:]
            activations = activations.reshape(-1, args.n_hid)
            activations = scaler.transform(activations)
            predictions = classifier.predict(activations)
            print(f"\n\nPreds: {predictions}\n\nTarget: {target.reshape(-1,)}\n\n")
            print(f"ERR: {np.square(predictions - target)}")
            preds.add(predictions)
            gts.add(target[0])
            mse = np.mean(np.square(predictions - target))
            rmse = np.sqrt(mse)
            norm = np.sqrt(np.square(target).mean())
            nrmse = rmse / (norm + 1e-9)
            return nrmse

        # print(train_dataset)
        # print("\n\n")
        dataset = train_dataset.to(device)
        # print(dataset)
        target = pd.DataFrame(train_target)[0].to_numpy()
        dataset = train_dataset.reshape(1, -1, 1).to(device)
        target = train_target.reshape(-1, 1).numpy()
        print(dataset)
        print("\n\n")
        print(target)

        print(f"\n\nTrain dataset: {(dataset.shape)}\nTarget: {(target.shape)}")
        # print("\n\n\n----------------------------")
        # print("[MAIN] before call")
        # print("----------------------------\n\n\n")
        activations = model(dataset)[0].cpu().numpy()
        WASHOUT = dataset[0].shape[0] - target.shape[0] # calculate difference in number of rows between the dataset and the target
        activations = activations[:, (WASHOUT):] # !!!!!!!!!!!!!!!!
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
        
        # print predictions and ground truth in a plot
        # plt.plot(preds, gts, label="Lorenz Attractor")
        # plt.xlabel("t")
        # plt.ylabel("Data")
        # plt.title("Lorenz Attractor")

    mean = np.mean(NRMSE)
    std = np.std(NRMSE)
    lastprint = ' ##################################################################### \n'
    lastprint += 'Mean NRMSE ' + str(mean) + ',    std ' + str(std) + '\n'
    lastprint += ' ##################################################################### \n'
    print(lastprint)

    f = open(f'{main_folder}/{namefile}.txt', 'a')
    f.write(lastprint)
    f.close()


