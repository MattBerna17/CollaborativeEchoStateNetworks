"""
porting of https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
in pytorch.

If you use this code in your work, please cite the following paper,
in which the concept of Deep Reservoir Computing has been introduced:

Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:
A  critical  experimental  analysis.    Neurocomputing268,  87–99  (2017).
https://doi.org/10.1016/j.neucom.2016.12.08924.
"""

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
                 leaky=1, connectivity_input=10, connectivity_recurrent=10,
                 feedback_size=0, neighbour_feedback_size=0, neighbour_scaling=1.):
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
        :param feedback_size: number of feedback connections from the output to the reservoir (for the teacher forcing technique)
        :param neighbour_feedback_size: number of connections between reservoirs (DISCLAIMER: for now, this is just a boolean parameter: 0 means "no connection between reservoirs" and n>1 means "fully connected reservoirs")
        :param neighbour_scaling: scaling factor for the connection matrix between reservoirs
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
        self.feedback_size = feedback_size
        self.neighbour_scaling = neighbour_scaling
        self.index = index
        self.n_layers = n_layers

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

        # if feedback is enabled, create the feedback kernel, which cannot be trained (statically set), just like any other kernel
        if feedback_size > 0:
            self.feedback_kernel = nn.Parameter(sparse_tensor_init(3, self.units, C=self.feedback_size) * (1 - (self.neighbour_scaling if self.neighbour_scaling > 0 else 0) - self.input_scaling), requires_grad=False) # equivalent to the W_fb in the original paper: its purpose is to multiply the previous time step's ground truth output to create a feedback loop.
        else:
            self.feedback_kernel = None # if feedback is not needed

        self.neighbour_feedback_size = neighbour_feedback_size
        if self.neighbour_feedback_size > 0:
            # create a neighbour kernel (W_nb) which is just an n_layers by 1 matrix which contains the connection weights between reservoirs.
            # the value W_nb[i] is just a scalar (defining the connection strength between current layer and layer i)
            # which will be multiplied by the i layer's expansion of the previous time step x_i(t-1).
            self.neighbour_kernel = nn.Parameter(create_sparse_connection_matrix(self.n_layers, 1.0) * self.neighbour_scaling, requires_grad=False)
            self.neighbour_kernel[index] = 0.0 # remove the self connection
            # print(f"[CELL {self.index}]\n{self.neighbour_kernel}\n\n")
        else:
            self.neighbour_kernel = None
    


    def forward(self, ut, x_prev, y_prev=None, X_neighbours=None):
        """ Computes the output of the cell given the input and previous state.

        :param ut: input at time step t (shape: [1, input_size])
        :param x_prev: previous state at time step t-1 (shape: [1, units])
        :param y_prev: previous ground truth at time step t-1 (shape: [1, input_size])
        :param X_neighbours: list of all previous reservoirs' state (shape: [n_layers, 1, units])
        :return: xt, xt
        """
        input_part = torch.mm(ut, self.kernel)
        state_part = torch.mm(x_prev, self.recurrent_kernel)

        if self.feedback_kernel is not None and y_prev is not None:
            feedback_part = torch.mm(y_prev, self.feedback_kernel) # multiply the feedback kernel with the previous output
        else:
            feedback_part = 0
        
        if self.neighbour_kernel is not None and X_neighbours is not None:
            neighbours_part = 0
            assert len(X_neighbours) == self.n_layers
            for i in range(self.n_layers): # for each neighbour (reservoir)
                if X_neighbours[i] is not None:
                    # add the neighbour's state (multiplied by the weight W_nb[n]) to the neighbours_part
                    neighbours_part += torch.mul(X_neighbours[i], self.neighbour_kernel[i]) # W_nb[n] * x_n(t) ---> scalar x vector(n_hid = self.units)
        else:
            neighbours_part = 0
                
        output = torch.tanh(input_part + self.bias + state_part + feedback_part + neighbours_part)
        leaky_output = x_prev * (1 - self.leaky) + output * self.leaky

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
                 leaky=1, connectivity_input=10, connectivity_recurrent=10, feedback_size=0, neighbour_feedback_size=0, neighbour_scaling=1.):
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
        :param feedback_size: size of the feedback kernel (teacher forcing)
        :param neighbour_feedback_size: size of the neighbour feedback kernel (same disclaimer as before, this is just a boolean parameter: 0 means "no connection between reservoirs" and n>1 means "fully connected reservoirs")
        :param neighbour_scaling: scaling factor for the connection matrix between reservoirs
        """
        super().__init__()
        self.net = ReservoirCell(input_size, units, index, n_layers, input_scaling,
                                 spectral_radius, leaky, connectivity_input,
                                 connectivity_recurrent, feedback_size=feedback_size, neighbour_feedback_size=neighbour_feedback_size, neighbour_scaling=neighbour_scaling)
        # self.net.save_parameters(f"./params/cell_{index}_params.pth") # save the parameters of the reservoir

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.net.units)

    def forward(self, x, y, h_prev=None, X_neighbours=None):
        """ Computes the output of the cell given the input and previous state.

        :param x:
        :param y: Y target tensor
        :param h_prev: h[0]
        :return: h, ht
        """

        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0]).to(x.device)

        hs = []
        for t in range(x.shape[1]):
            xt = x[0, t, :].reshape(-1, self.net.input_size)
            y_prev = torch.Tensor(y[t-1].reshape(-1, 3)) if t > 0 and y is not None else None
            _, h_prev = self.net(xt, h_prev, y_prev=y_prev, X_neighbours=X_neighbours) # call to forward method
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev












class DeepReservoir(torch.nn.Module):
    def __init__(self, input_size=1, tot_units=100, n_layers=1, concat=False,
                 input_scaling=1, inter_scaling=1,
                 spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=10,
                 connectivity_input=10,
                 connectivity_inter=10,
                 feedback_size=0,
                 neighbour_feedback_size=0,
                 neighbour_scaling=1.
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
        :param feedback_size: size of the feedback matrix
        :param neighbour_feedback_size: size of the neighbour feedback matrix
        :param neighbour_scaling: scaling factor for the connection matrix between reservoirs
        """
        super().__init__()
        number_of_reservoirs = n_layers # TODO: change this!!!!
        self.n_layers = n_layers
        self.tot_units = tot_units
        self.concat = concat
        self.batch_first = True  # DeepReservoir only supports batch_first

        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter
        self.neighbour_feedback_size = neighbour_feedback_size

        self.reservoirs_connection_matrix = create_sparse_connection_matrix(number_of_reservoirs, 0.5) # TODO: change connectivity using a new parameter
        self.input_size = input_size
        self.neighbour_scaling = neighbour_scaling


        if concat:
            self.layers_units = int(tot_units / n_layers)
            # create the first reservoir:
            reservoir_layers = [
                ReservoirLayer(
                    input_size=input_size,
                    units=self.layers_units + tot_units % n_layers,
                    index=0,
                    n_layers=n_layers,
                    input_scaling=input_scaling,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_1,
                    connectivity_recurrent=connectivity_recurrent,
                    feedback_size=feedback_size,
                    neighbour_feedback_size=neighbour_feedback_size,
                    neighbour_scaling=neighbour_scaling
                )
            ]
            # last_h_size may be different for the first layer
            # because of the remainder if concat=True
            last_h_size = self.layers_units + tot_units % n_layers
        else:
            self.layers_units = tot_units
            # create the first reservoir:
            reservoir_layers = [
                ReservoirLayer(
                    input_size=input_size,
                    units=self.layers_units,
                    index=0, 
                    n_layers=n_layers,
                    input_scaling=input_scaling,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_1,
                    connectivity_recurrent=connectivity_recurrent,
                    feedback_size=feedback_size,
                    neighbour_feedback_size=neighbour_feedback_size,
                    neighbour_scaling=neighbour_scaling
                )
            ]
            last_h_size = self.layers_units

        # create all the other reservoirs:
        for i in range(1, n_layers):
            reservoir_layers.append(ReservoirLayer(
                input_size=input_size,
                units=self.layers_units,
                index=i,
                n_layers=n_layers,
                input_scaling=input_scaling_others,
                spectral_radius=spectral_radius,
                leaky=leaky,
                connectivity_input=connectivity_input_others,
                connectivity_recurrent=connectivity_recurrent,
                feedback_size=feedback_size,
                neighbour_feedback_size=neighbour_feedback_size,
                neighbour_scaling=neighbour_scaling
            ))
            last_h_size = self.layers_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    def forward(self, X_dataset, Y=None):
        """ compute the output of the deep reservoir.

        :param X:
        :return: hidden states (B, T, F), last state (L, B, F)
        """
        states = []  # list of all the states in all the layers
        states_last = []  # list of the states in all the layers for the last time step
        # states_last is a list because different layers may have different size.
        X = None
        h_lasts = [None for _ in range(self.n_layers)]
        for res_idx, res_layer in enumerate(self.reservoir):
            # right now, i pass the input to each layer, and also pass the states of the previous layers as neighbour values to the current layer.

            [X, h_last] = res_layer(X_dataset, y=Y, X_neighbours=h_lasts)
            h_lasts[res_idx] = h_last
            
            states.append(X)
            states_last.append(h_last)

        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
        return states, states_last

