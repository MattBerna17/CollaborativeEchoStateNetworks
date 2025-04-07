"""
porting of https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
in pytorch.

If you use this code in your work, please cite the following paper,
in which the concept of Deep Reservoir Computing has been introduced:

Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:
A  critical  experimental  analysis.    Neurocomputing268,  87–99  (2017).
https://doi.org/10.1016/j.neucom.2016.12.08924.
"""

from sklearn import preprocessing
from sklearn.linear_model import Ridge
import torch
from torch import nn
import numpy as np
from utils import create_sparse_connection_matrix

torch.manual_seed(42)
np.random.seed(42)




def sparse_eye_init(M: int) -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse identity matrix for the
    re-scaling of the sparse recurrent kernel in presence of non-zero leakage.
    The neurons are connected according to a ring topology, where each neuron
    receives input only from one neuron and propagates its activation only to
    one other neuron. All the non-zero elements are set to 1.

    :param M: number of hidden units
    :return: dense weight matrix
    """
    dense_shape = torch.Size([M, M])

    # gives the shape of a ring matrix:
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, :] = i
    values = torch.ones(M)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """ Generates an M x N matrix to be used as sparse (input) kernel
    For each row only C elements are non-zero (i.e., each input dimension is
    projected only to C neurons). The non-zero elements are generated randomly
    from a uniform distribution in [-1,1]

    :param M: number of rows
    :param N: number of columns
    :param C: number of nonzero elements
    :return: MxN dense matrix
    """
    dense_shape = torch.Size([M, N])  # shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k = k + 1
    #values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    values = 2 * np.random.rand(M * C).astype('f') - 1
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_recurrent_tensor_init(M: int, C: int = 1) -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse recurrent kernel.
    For each column only C elements are non-zero (i.e., each recurrent neuron
    take sinput from C other recurrent neurons). The non-zero elements are
    generated randomly from a uniform distribution in [-1,1].

    :param M: number of hidden units
    :param C: number of nonzero elements
    :return: MxM dense matrix
    """
    assert M >= C
    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k = k + 1
    #values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    values = 2 * np.random.rand(M * C).astype('f') - 1
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales W to have rho(W) = rho_desired

    :param W:
    :param rho_desired:
    :return:
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)




class ReservoirCell(torch.nn.Module):
    def __init__(self, input_size, units, index, n_layers=1, input_scaling=1., spectral_radius=0.99,
                 leaky=1, connectivity_input=10, connectivity_recurrent=10):
        """ Shallow reservoir to be used as cell of a Recurrent Neural Network.

        :param input_size: number of input units
        :param units: number of recurrent neurons in the reservoir
        :param index: index of the reservoir in the ReservoirLayer list of the DeepReservoir
        :param n_layers: number of layers created in the DeepReservoir
        :param input_scaling: max abs value of a weight in the input-reservoir
            connections. Note that whis value also scales the unitary input bias
        :param spectral_radius: max abs eigenvalue of the recurrent matrix
        :param leaky: leaking rate constant of the reservoir ([0, 1] how to weight the previous state is 1-leaky)
        :param connectivity_input: number of outgoing connections from each
            input unit to the reservoir
        :param connectivity_recurrent: number of incoming recurrent connections
            for each reservoir unit
        """
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent

        print(f"[RESERVOIR CELL] created with {self.units} units.\n")

        self.kernel = sparse_tensor_init(input_size, self.units,
                                         self.connectivity_input) * self.input_scaling
        self.kernel = nn.Parameter(self.kernel, requires_grad=False) # equivalent to the W_in in the original paper: its purpose is to multiply the input tensor to create a higher dimension representation of the input itself.

        W = sparse_recurrent_tensor_init(self.units, C=self.connectivity_recurrent)
        # re-scale the weight matrix to control the effective spectral radius
        # of the linearized system
        if self.leaky == 1:
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = W
        else:
            I = sparse_eye_init(self.units)
            W = W * self.leaky + (I * (1 - self.leaky))
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = (W + I * (self.leaky - 1)) * (1 / self.leaky)
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)

        self.bias = (torch.rand(self.units) * 2 - 1)
        self.bias = nn.Parameter(self.bias, requires_grad=False)
    


    def forward(self, ut, h_prev):
        """ Computes the output of the cell given the input and previous state.

        :param ut: input at time step t (shape: [1, input_size])
        :param h_prev: previous state of the reservoir
        :return: xt, xt
        """
        input_part = torch.mm(ut, self.kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)
        output = torch.tanh(input_part + self.bias + state_part)
        leaky_output = h_prev * (1 - self.leaky) + output * self.leaky

        return leaky_output, leaky_output
    
    

    # helper functions for reproducibility of experiments
    def save_parameters(self, filename):
        """
        Save the parameters of the ReservoirCell to a file.

        Args:
            filename (str): The filename to save the parameters to.
        """
        torch.save({
            'kernel': self.kernel,
            'recurrent_kernel': self.recurrent_kernel,
            'bias': self.bias
        }, filename)

    def set_parameters(self, filename):
        """
        Set the parameters of the ReservoirCell from a file.

        Args:
            filename (str): The filename to load the parameters from.
        """
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        self.kernel = checkpoint['kernel']
        self.recurrent_kernel = checkpoint['recurrent_kernel']
        self.bias = checkpoint['bias']
        self.kernel.requires_grad = False
        self.recurrent_kernel.requires_grad = False
        self.bias.requires_grad = False












class ReservoirLayer(torch.nn.Module):
    def __init__(self, input_size, units, index, n_layers=1, input_scaling=1., spectral_radius=0.99,
                 leaky=1, connectivity_input=10, connectivity_recurrent=10):
        """ Shallow reservoir to be used as Recurrent Neural Network layer.

        :param input_size: number of input units
        :param units: number of recurrent neurons in the reservoir
        :param index: index of the layer (reservoir)
        :param n_layers: number of layers (reservoirs) in the deep reservoir
        :param input_scaling: max abs value of a weight in the input-reservoir
            connections. Note that whis value also scales the unitary input bias
        :param spectral_radius: max abs eigenvalue of the recurrent matrix
        :param leaky: leaking rate constant of the reservoir
        :param connectivity_input: number of outgoing connections from each
            input unit to the reservoir
        :param connectivity_recurrent: number of incoming recurrent connections
            for each reservoir unit
        """
        super().__init__()
        self.net = ReservoirCell(input_size, units, index, n_layers, input_scaling,
                                 spectral_radius, leaky, connectivity_input,
                                 connectivity_recurrent)
        # self.net.save_parameters(f"./params/cell_{index}_params.pth") # save the parameters of the reservoir

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.net.units)

    def forward(self, u, h_prev=None):
        """ Computes the output of the cell given the input and previous state.

        :param u: dataset
        :param h_prev: previous state of the reservoir
        :return: h, ht next state of the reservoir
        """

        if h_prev is None:
            h_prev = self.init_hidden(u.shape[0]).to(u.device)

        hs = []
        # print(u.shape)
        for t in range(u.shape[1]):
            ut = u[0, t, :].reshape(-1, self.net.input_size)
            _, h_prev = self.net(ut, h_prev=h_prev) # call to forward method
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev
    

    def train(self, U, Y, washout=200, solver=None, regul=1e-3):
        """
        Function to train the deep reservoir's readout layer (W_out matrix) given the dataset and the target.

        :param U: dataset
        :param Y: target
        :param washout: number of elements to remove from the beginning of the dataset
        :param solver: solver of the ridge regression
        :param regul: regularization coefficient
        :return scaler: scaler used to scale the activations
        :return classifier: trained readout layer
        """
        activations = self(U)[0].cpu().numpy() # train the reservoir layer to get the activations (activations = hs)
        activations = activations.reshape(-1, self.net.units) # reshape the activations to torch.Size([rows=len(train_dataset), columns=args.n_hid])
        activations = activations[washout:] # remove washout elements
        scaler = preprocessing.StandardScaler().fit(activations) # train the scaler on the activations
        self.scaler = scaler
        
        # do not save the activations AFTER the scaler's application
        self.activations = activations # save the activations before scaling them (for each iteration in the test method, the activations are scaled with the same scaler)

        # print(f"Activations max and min: {activations.max()} - {activations.min()}\n")
        activations = scaler.transform(activations) # scale the activations
        # print(f"Scaled activations max and min: {activations.max()} - {activations.min()}\n\n")
        # save_matrix_to_file(activations, "train_activations")

        print(f"\n\nConditioning: {np.linalg.cond(activations)}\n\n")

        target = Y[washout:] # remove first washout elements
        if solver is None:
            classifier = Ridge(alpha=regul, max_iter=1000).fit(activations, target)
        elif solver == 'svd':
            classifier = Ridge(alpha=regul, solver='svd').fit(activations, target)
        else:
            classifier = Ridge(alpha=regul, solver=solver).fit(activations, target)
        self.classifier = classifier

        return scaler, classifier

    def predict(self, n_iter):
        """
        Function to predict the next n_iter states of the dynamic system

        :param n_iter: number of iterations to predict
        :return: predictions
        """
        activations = torch.tensor(self.activations, dtype=torch.float32)
        # take the last training iteration's activations and predict the next state of the system (so, at time step dim_training + 1, i.e. first element in the validation/test dataset)
        scaled_activations = self.scaler.transform(activations)
        # print(f"scaled activations dimensions: {scaled_activations.shape}\n\n")
        ot = torch.tensor(self.classifier.predict(scaled_activations)[-1].reshape(1, 1, self.net.input_size), dtype=torch.float32)
        # print(f"First prediction: {ot}\n")
        predictions = [] # to store all the predictions
        # use n_iter+1 because the first guess is the o(t-1), the first element in the dataset, so the second iteration predicts the first target
        for i in range(n_iter+1): # for each timestep
            # get new activations from the forward method, passing the previous activations and the prediction of the past iteration (i.e. h(t-1) and o(t-1))
            new_activation = self(ot, activations[-1].reshape(1, -1))[0].reshape(1, -1)
            activations = torch.cat((activations, new_activation), dim=0) # add the new activation to the activations
            self.activations = activations.numpy() # save the updated activations in numpy form
            scaled_activations = self.scaler.transform(activations) # scale the activations together (to avoid scaling the past activations multiple times)
            scaled_activations = torch.tensor(scaled_activations, dtype=torch.float32)
            # print(f"Scaled activations dimension: {scaled_activations.shape}")
            ot = torch.tensor(self.classifier.predict(scaled_activations[-1].unsqueeze(0)).reshape(1, 1, self.net.input_size), dtype=torch.float32) # predict the next state (o(t))
            predictions.append(ot)
        # self.activations = activations.numpy() # check if the type of activations is numpy ndarray first
        return predictions[1:]












class DeepReservoir(torch.nn.Module):
    def __init__(self, input_size=1, tot_units=100, n_layers=1, concat=False,
                 input_scaling=1, inter_scaling=1,
                 spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=10,
                 connectivity_input=10,
                 connectivity_inter=10
                ):
        """ Deep Reservoir layer.
        The implementation realizes a number of stacked RNN layers using the
        ReservoirCell as core cell. All the reservoir layers share the same
        hyper-parameter values (i.e., same number of recurrent neurons, spectral
        radius, etc. ).

        :param input_size: dimension of the input
        :param tot_units: number of recurrent units.
            if concat == True this is the total number of units
            if concat == False this is the number of units for each
                reservoir level
        :param n_layers: number of layers (reservoirs)
        :param concat: if True the returned state is given by the
            concatenation of all the states in the reservoir levels
        :param input_scaling: scaling coeff. of the first reservoir layer
        :param inter_scaling: scaling coeff. of all the other levels (> 1)
        :param spectral_radius:
        :param leaky: leakage coefficient of all levels
        :param connectivity_recurrent:
        :param connectivity_input: input connectivity coefficient of the input weight matrix
        :param connectivity_inter: input connectivity coefficient of all the inter-levels weight matrices
        """
        super().__init__()
        self.n_layers = n_layers
        self.tot_units = tot_units
        self.concat = concat
        self.batch_first = True  # DeepReservoir only supports batch_first

        input_scaling_others = inter_scaling
        connectivity_input_1 = tot_units // n_layers
        connectivity_input_others = tot_units // n_layers
        connectivity_recurrent = tot_units // n_layers


        if concat:
            self.layers_units = int(tot_units / n_layers)
            # create the first reservoir:
            reservoir_layers = [
                ReservoirLayer(
                    input_size=(input_size//n_layers),
                    units=self.layers_units + tot_units % n_layers,
                    index=0,
                    n_layers=n_layers,
                    input_scaling=input_scaling,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_1,
                    connectivity_recurrent=connectivity_recurrent
                )
            ]
            # last_h_size may be different for the first layer
            # because of the remainder if concat=True
            last_h_size = self.layers_units + tot_units % n_layers
        else:
            self.layers_units = tot_units // n_layers
            # create the first reservoir:
            reservoir_layers = [
                ReservoirLayer(
                    input_size=(input_size//n_layers),
                    units=self.layers_units,
                    index=0, 
                    n_layers=n_layers,
                    input_scaling=input_scaling,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_1,
                    connectivity_recurrent=connectivity_recurrent
                )
            ]
            last_h_size = self.layers_units

        # create all the other reservoirs:
        for i in range(1, n_layers):
            reservoir_layers.append(ReservoirLayer(
                input_size=(input_size//n_layers),
                units=self.layers_units,
                index=i,
                n_layers=n_layers,
                input_scaling=input_scaling_others,
                spectral_radius=spectral_radius,
                leaky=leaky,
                connectivity_input=connectivity_input_others,
                connectivity_recurrent=connectivity_recurrent
            ))
            last_h_size = self.layers_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    

    def forward(self, U, h_prev=None):
        """ compute the output of the deep reservoir.

        :param X:
        :return: hidden states (B, T, F), last state (L, B, F)
        """
        states = []  # list of all the states in all the layers
        states_last = []  # list of the states in all the layers for the last time step
        if h_prev is not None:
            states.append(h_prev)
        # states_last is a list because different layers may have different size.
        for res_idx, res_layer in enumerate(self.reservoir):
            # right now, i pass the input to each layer, and also pass the states of the previous layers as neighbour values to the current layer.

            [H, h_last] = res_layer(U, h_prev=h_prev)
            
            states.append(H)
            states_last.append(h_last)

        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
        states_last = states_last[-1]
        return states, states_last
    

    def train(self, U, Y, washout=200, solver=None, regul=1e-3):
        """
        Function to train the deep reservoir's readout layer (W_out matrix) given the dataset and the target.

        :param U: dataset
        :param Y: target
        :param washout: number of elements to remove from the beginning of the dataset
        :param solver: solver of the ridge regression
        :param regul: regularization coefficient
        :return scaler: scaler used to scale the activations
        :return classifier: trained readout layer
        """
        classifiers = []
        scalers = []
        if self.n_layers > 1:
            for l in range(self.n_layers):
                U_layer = U[:, :, l].reshape(1, -1, 1) # shape = [1, n_examples, 1]
                Y_layer = Y[:, (l+1)%Y.shape[1]].reshape(-1, 1) # shape = [n_examples, 1]
                scaler, classifier = self.reservoir[l].train(U_layer, Y_layer, washout, solver, regul)
                scalers.append(scaler)
                classifiers.append(classifier)
        else:
            scaler, classifier = self.reservoir[0].train(U, Y, washout, solver, regul)
            scalers.append(scaler)
            classifiers.append(classifier)
        
        return scalers, classifiers

    def predict(self, n_iter, y_init, Y):
        """
        Function to predict the next n_iter states of the dynamic system

        :param n_iter: number of iterations to predict
        :return: predictions
        """
        if self.n_layers > 1:
            # initialize first ot
            # separate ot in 3 dims (x, y, z)
            # predict for each layer the respective dimension
            layers_activations = [torch.tensor(self.reservoir[l].activations, dtype=torch.float32) for l in range(self.n_layers)]
            layers_scaled_activations = [self.reservoir[l].scaler.transform(layers_activations[l]) for l in range(self.n_layers)]
            ot_unordered = [
                torch.tensor(
                    self.reservoir[l].classifier.predict(
                        layers_scaled_activations[l]
                    )[-1].reshape(
                        1, 1, self.reservoir[l].net.input_size
                    ), dtype=torch.float32
                )
                for l in range(self.n_layers)
            ]
            # ot_unordered.append(torch.tensor(y_init[0]).reshape(1, 1, 1)) # add the first element of the target to the last layer
            ot = torch.stack((ot_unordered[-1:] + ot_unordered[0:-1])) # reorder elements based on shift
            # print(f"ot shape: {ot.shape}\n")
            # ot = torch.tensor(y_init, dtype=torch.float32).reshape(3, 1, 1, 1)
            # print(f"ot ordered: {ot}\n\n\n")

            predictions = []
            for i in range(n_iter):
                ot_unordered = [None for _ in range(self.n_layers)]

                # ot_unordered[self.n_layers-1] = torch.tensor(Y[i, 0]).reshape(1, 1, 1)

                for l in range(self.n_layers):
                    new_layer_activation = self.reservoir[l](ot[l], layers_activations[l][-1].reshape(1, -1))[0].reshape(1, -1) # expand input of the layer and obtain the new h(t)
                    # print(f"[R {l}] scaled new layer activation: {self.reservoir[l].scaler.transform(new_layer_activation)}\n")
                    # print(f"layers activations: {layers_activations[l]}\n")
                    layers_activations[l] = torch.cat((layers_activations[l], new_layer_activation), dim=0)
                    self.reservoir[l].activations = layers_activations[l].numpy()
                    layers_scaled_activations[l] = self.reservoir[l].scaler.transform(layers_activations[l])
                    # print(f"layers activations transformed: {layers_scaled_activations[l]}\n\n")
                    layers_scaled_activations[l] = torch.tensor(layers_scaled_activations[l], dtype=torch.float32)
                    ot_unordered[l] = torch.tensor(self.reservoir[l].classifier.predict(layers_scaled_activations[l][-1].unsqueeze(0)).reshape(1, 1, self.reservoir[l].net.input_size), dtype=torch.float32)
                # print("\n")
                # exit(0)
                ot = torch.stack((ot_unordered[-1:] + ot_unordered[0:-1]))
                print(f"ot: {ot}\n")
                predictions.append(ot)

            return predictions
        else:
            return self.reservoir[0].predict(n_iter)