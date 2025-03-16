import datetime
from scipy.integrate import odeint
import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch import nn
# from esn_alternative import spectral_norm_scaling
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random


# torch.manual_seed(42)
# np.random.seed(42)


def count_parameters(model):
    """Return total number of parameters and
    trainable parameters of a PyTorch model.
    """
    params = []
    trainable_params = []
    for p in model.parameters():
        params.append(p.numel())
        if p.requires_grad:
            trainable_params.append(p.numel())
    pytorch_total_params = sum(params)
    pytorch_total_trainableparams = sum(trainable_params)
    print('Total params:', pytorch_total_params)
    print('Total trainable params:', pytorch_total_trainableparams)


def n_params(model):
    """Return total number of parameters of the
    LinearRegression model of Scikit-Learn.
    """
    return (sum([a.size for a in model.coef_]) +
            sum([a.size for a in model.intercept_]))


# ########## Torch Dataset for FordA ############## #
import torch.utils.data as data
import torch.nn.functional as F

class datasetforRC(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding gives problems with scikit-learn LogisticRegression of RC models
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)


class FordA_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=2).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)


class Adiac_dataset(data.Dataset):
    """
    This class assumes mydata to have the form:
            [ (x1,y1), (x2,y2) ]
    where xi are inputs, and yi are targets.
    """

    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp, idx_targ = torch.Tensor(idx_inp), torch.Tensor([idx_targ])
        # reshape time series for torch (batch, inplength, inpdim)
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        # one-hot encoding targets
        idx_targ = F.one_hot(idx_targ.type(torch.int64), num_classes=37).float()
        # reshape target for torch (batch, classes)
        idx_targ = idx_targ.reshape(idx_targ.shape[1])
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)
# ################################################# #


class LSTM(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_inp, n_hid, batch_first=True,
                                  num_layers=1)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x):
        out, h = self.lstm(x)
        out = self.readout(out[:, -1])
        return out



class RNN_Separate(nn.Module):
    def __init__(self, n_inp, n_hid):
        super().__init__()
        self.i2h = torch.nn.Linear(n_inp, n_hid)
        self.h2h = torch.nn.Linear(n_hid, n_hid)
        self.n_hid = n_hid

    def forward(self, x):
        states = []
        state = torch.zeros(x.size(0), self.n_hid, requires_grad=False).to(x.device)
        for t in range(x.size(1)):
            state = torch.tanh(self.i2h(x[:, t])) + torch.tanh(self.h2h(state))
            states.append(state)
        return torch.stack(states, dim=1), state

class RNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, separate_nonlin=False):
        super().__init__()
        if separate_nonlin:
            self.rnn = RNN_Separate(n_inp, n_hid)
        else:
            self.rnn = torch.nn.RNN(n_inp, n_hid, batch_first=True,
                                    num_layers=1)
        self.readout = torch.nn.Linear(n_hid, n_out)

    def forward(self, x):
        out, h = self.rnn(x)
        out = self.readout(out[:, -1])
        return out



def get_cifar_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [47000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=bs_test,
                                               shuffle=False,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False,
                                              drop_last=True)

    return train_loader, valid_loader, test_loader



def get_lorenz_attractor(lag=1, washout=200):
    dataset = pd.read_csv("lorenz.csv", index_col="t").drop(columns=["Unnamed: 0"])
    dataset = torch.tensor(dataset.values).float()
    (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target) = separate_training_validation_test(dataset, washout, lag)
    # print(train_dataset)
    # print(train_target)
    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)





def get_lorenz(N=3, F=8, num_batch=128, lag=25, washout=200, window_size=0, serieslen=20):
    # https://en.wikipedia.org/wiki/Lorenz_96_model
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    dt = 0.01
    t = np.arange(0.0, serieslen+(lag*dt)+(washout*dt), dt)
    dataset = []
    for i in range(num_batch):
        x0 = np.random.rand(N) + F - 0.5 # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)
    dataset = np.stack(dataset, axis=0)
    dataset = torch.from_numpy(dataset).float()

    if window_size > 0:
        windows, targets = [], []
        for i in range(dataset.shape[0]):
            w, t = get_fixed_length_windows(dataset[i], window_size, prediction_lag=lag)
        windows.append(w)
        targets.append(t)
        return torch.utils.data.TensorDataset(torch.cat(windows, dim=0), torch.cat(targets, dim=0))
    else:
        return separate_training_validation_test(dataset)
    
def separate_training_validation_test(dataset: torch.Tensor, washout=200, lag=1):
    end_train = int(dataset.shape[0] / 2)
    end_val = end_train + int(dataset.shape[0] / 4)
    end_test = dataset.shape[0]

    # ! TODO: check and correct washout offset (ask gallicchio)
    train_dataset = dataset[:end_train-lag]
    train_target = dataset[lag:end_train]

    val_dataset = dataset[end_train:end_val-lag]
    val_target = dataset[end_train+lag:end_val]

    test_dataset = dataset[end_val:end_test-lag]
    test_target = dataset[end_val+lag:end_test]

    # print(f"\nDimensioni:\ntrain dataset: {(train_dataset.shape)}")
    # print(f"train target: {(train_target.shape)}")
    # print(f"val dataset: {(val_dataset.shape)}")
    # print(f"val target: {(val_target.shape)}")
    # print(f"test dataset: {(test_dataset.shape)}")
    # print(f"test target: {(test_target.shape)}\n")

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)





def create_sparse_connection_matrix(number_of_reservoirs, connection_density) -> torch.Tensor:
    """
    Create a sparse connection matrix between the reservoirs, diagonal elements are zero
    :param: number_of_reservoirs: number of reservoirs
    :param: connection_density: density of connections (percentage of non-zero elements)
    """
    connectivity = torch.zeros((number_of_reservoirs, 1))
    for i in range(number_of_reservoirs):
        if np.random.rand() < connection_density:
            connectivity[i] = np.random.rand()
    return connectivity







def save_matrix_to_file(matrix: torch.Tensor, filename):
    now = datetime.datetime.now().strftime("%m_%d__%H_%M")
    now_file = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    # create folder with datetime expressed in now if it doesn't exist
    if not os.path.exists('./data/' + now):
        os.makedirs('./data/' + now)
    filename = './data/' + now + '/' + now_file + '___' + filename + '.txt'
    
    with open(filename, 'w') as f:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                f.write(str(matrix[i][j].item()) + ',')
            f.write('\n')










def plot_lorenz_attractor_with_error(predictions: torch.Tensor, targets: torch.Tensor, title):
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.set_title('Lorenz Attractor')
    fig.canvas.manager.set_window_title(title)
    x_target = targets[:, 0]
    y_target = targets[:, 1]
    z_target = targets[:, 2]
    x_pred = predictions[:, 0]
    y_pred = predictions[:, 1]
    z_pred = predictions[:, 2]
    ax.plot3D(x_pred, y_pred, z_pred, 'blue', label='Predictions')
    ax.plot3D(x_target, y_target, z_target, 'red', label='Targets')
    ax.set_title('3D Parametric Plot')

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    plt.show()
    return


def get_mackey_glass(lag=1, washout=200, window_size=0):
    """
    Predict the next lag-th item of mackey-glass series
    """
    with open('mackey-glass.csv', 'r') as f:
        dataset = f.readlines()[0]  # single line file

    # 10k steps
    dataset = torch.tensor([float(el) for el in dataset.split(',')]).float()

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
        train_target = dataset[washout+lag:end_train]

        val_dataset = dataset[end_train:end_val-lag]
        val_target = dataset[end_train+washout+lag:end_val]

        test_dataset = dataset[end_val:end_test-lag]
        test_target = dataset[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_narma10(lag=0, washout=200):
    """
    Predict the output of a narma10 series
    """
    with open('narma10.csv', 'r') as f:
        dataset = f.readlines()[0:2]  # 2 lines file

    # 10k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    end_train = int(dataset[0].shape[0] / 2)
    end_val = end_train + int(dataset[0].shape[0] / 4)
    end_test = dataset[0].shape[0]

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_narma30(lag=0, washout=200):
    """
    Predict the output of a narma30 series
    """
    with open('narma30.csv', 'r') as f:
        dataset = f.readlines()[0:2]  # 2 lines file

    # 10k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    end_train = int(dataset[0].shape[0] / 2)
    end_val = end_train + int(dataset[0].shape[0] / 4)
    end_test = dataset[0].shape[0]

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_piSineDelay10(lag=0, washout=200, ergodic=False):
    """
    Predict the output of a sin(pi*u[t-10]) series
    given in input u[t]
    """
    if ergodic:
        with open('SineDelay10.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file
    else:
        with open('piSineDelay10.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file

    # 6k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    # firs 4k training, then 1k validation, and 1k test
    end_train = 4000
    end_val = end_train + 1000
    end_test = 6000

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_ctXOR(delay=2, washout=200, lag=0):
    """
    Predict the output of a sign(r[t])*abs(r[t])^degree series
    where r[t] = u[t-delay]*u[t-delay-1]
    """
    if delay==5:
        with open('ctXOR_delay5_degree2.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file
    elif delay==15:
        with open('ctXOR_delay15_degree10.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file
    elif delay==10:
        with open('ctXOR_delay10_degree2.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file      
    elif delay==2:
        with open('ctXOR_delay2_degree2.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file      
    else:
        raise ValueError('Only delays 5, 10, or 15 available.')

    # 6k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    # firs 4k training, then 1k validation, and 1k test
    end_train = 4000
    end_val = end_train + 1000
    end_test = 6000

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_mnist_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [57000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader



def get_mnist_testing_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)
    
    return train_loader, test_loader



def load_har(root):
    """
    Dataset preprocessing code adapted from
    https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LSTM.ipynb
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]
    """
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]
    # FROM LABELS IDX (starting from 1) TO BINARY CLASSES (0-1)
    CLASS_MAP = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0}
    TRAIN = "train"
    TEST = "test"

    def load_X(X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            with open(signal_type_path, 'r') as file:
                X_signals.append(
                    [np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ', ' ').strip().split(' ') for row in file
                    ]]
                )

        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(y_path):
        with open(y_path, 'r') as file:
            y_ = np.array(
                [CLASS_MAP[int(row)] for row in file],
                dtype=np.int32
            )
        return y_


    X_train_signals_paths = [
        os.path.join(root, TRAIN, "Inertial Signals", signal+"train.txt") for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        os.path.join(root, TEST, "Inertial Signals", signal+"test.txt") for signal in INPUT_SIGNAL_TYPES
    ]

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(os.path.join(root, TRAIN, "y_train.txt"))
    y_test = load_y(os.path.join(root, TEST, "y_test.txt"))

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    val_length = int(len(train_dataset) * 0.3)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-val_length, val_length])
    return train_dataset, val_dataset, test_dataset


def get_fixed_length_windows(tensor, length, prediction_lag=1):
    assert len(tensor.shape) <= 2
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(-1)

    windows = tensor[:-prediction_lag].unfold(0, length, 1)
    windows = windows.permute(0, 2, 1)

    targets = tensor[length+prediction_lag-1:]
    return windows, targets  # input (B, L, I), target, (B, I)

