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
    def __init__(self, input_size, output_size, units, index, n_modules=1, input_scaling=1., spectral_radius=0.99, leaky=1, connectivity_input=10, connectivity_recurrent=10):
        """ Shallow reservoir to be used as cell of a Recurrent Neural Network.

        :param input_size: number of input units
        :param units: number of recurrent neurons in the reservoir
        :param index: index of the reservoir in the ReservoirModule list of the DeepReservoir
        :param n_modules: number of modules created in the DeepReservoir
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

        self.index = index
        self.input_size = input_size
        self.output_size = output_size
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent
        self.verbose = True

        print(f"[RESERVOIR CELL {self.index}] created with {self.units} units.\n")

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

        if self.verbose:
            print(f"[RESERVOIR CELL {self.index}] input: {ut}\n")
            print(f"[RESERVOIR CELL {self.index}] h_prev: {torch.median(h_prev)}\n")
            print(f"[RESERVOIR CELL {self.index}] input_part: {torch.median(input_part)}\n")
            print(f"[RESERVOIR CELL {self.index}] state_part: {torch.median(state_part)}\n")
            print(f"[RESERVOIR CELL {self.index}] output: {torch.median(output)}\n")
            print(f"[RESERVOIR CELL {self.index}] leaky_output: {torch.median(leaky_output)}\n\n\n")

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












class ReservoirModule(torch.nn.Module):
    def __init__(self, input_size, output_size, units, index, solver=None, regul=1e-3, n_modules=1, input_scaling=1., spectral_radius=0.99, leaky=1, connectivity_input=10, connectivity_recurrent=10):
        """ Shallow reservoir to be used as Recurrent Neural Network module.

        :param input_size: number of input units
        :param units: number of recurrent neurons in the reservoir
        :param index: index of the module (reservoir)
        :param n_modules: number of modules (reservoirs) in the deep reservoir
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
        self.net = ReservoirCell(input_size, output_size, units, index, n_modules, input_scaling,
                                 spectral_radius, leaky, connectivity_input,
                                 connectivity_recurrent)
        self.solver = solver
        self.regul = regul
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
        for t in range(u.shape[1]):
            # print(f"TIMESTEP {t}")
            ut = u[0, t, :].reshape(-1, self.net.input_size)
            _, h_prev = self.net(ut, h_prev=h_prev) # call to forward method
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev
    

    def fit(self, U, Y, washout=200):
        """
        Function to train the reservoir's readout module (W_out matrix) given the dataset and the target.

        :param U: dataset
        :param Y: target
        :param washout: number of elements to remove from the beginning of the dataset
        :param solver: solver of the ridge regression
        :param regul: regularization coefficient
        :return scaler: scaler used to scale the activations
        :return classifier: trained readout module
        """
        activations = self(U)[0].cpu().numpy() # train the reservoir module to get the activations (activations = hs)
        activations = activations.reshape(-1, self.net.units) # reshape the activations to torch.Size([rows=len(train_dataset), columns=args.n_hid])
        activations = activations[washout:] # remove washout elements
        scaler = preprocessing.StandardScaler().fit(activations) # train the scaler on the activations
        self.scaler = scaler
        self.activations = activations # save the activations before scaling them (for each iteration in the test method, the activations are scaled with the same scaler)
        activations = self.scaler.transform(self.activations) # scale the activations
        # print(f"\n\nConditioning: {np.linalg.cond(activations)}\n\n")
        target = Y[washout:] # remove first washout elements

        if self.solver is None:
            classifier = Ridge(alpha=self.regul, max_iter=1000).fit(activations, target)
        elif self.solver == 'svd':
            classifier = Ridge(alpha=self.regul, solver='svd').fit(activations, target)
        else:
            classifier = Ridge(alpha=self.regul, solver=self.solver).fit(activations, target)
        
        self.classifier = classifier

    def predict(self, n_iter):
        """
        Function to predict the next n_iter states of the dynamic system

        :param n_iter: number of iterations to predict
        :return: predictions
        """
        activations = torch.tensor(self.activations, dtype=torch.float32)
        scaled_activations = self.scaler.transform(activations)
        ot = torch.tensor(self.classifier.predict(scaled_activations)[-1].reshape(1, 1, self.net.input_size), dtype=torch.float32)
        predictions = [] # to store all the predictions
        for i in range(n_iter): # for each timestep
            # get new activations from the forward method, passing the previous activations and the prediction of the past iteration (i.e. h(t-1) and o(t-1))
            new_activation = self(ot, activations[-1].reshape(1, -1))[0].reshape(1, -1)
            activations = torch.cat((activations, new_activation), dim=0) # add the new activation to the activations
            self.activations = activations.numpy() # save the updated activations in numpy form
            scaled_activations = self.scaler.transform(activations) # scale the activations together (to avoid scaling the past activations multiple times)
            scaled_activations = torch.tensor(scaled_activations, dtype=torch.float32)
            ot = torch.tensor(self.classifier.predict(scaled_activations[-1].unsqueeze(0)).reshape(1, 1, self.net.input_size), dtype=torch.float32) # predict the next state (o(t))
            predictions.append(ot)
        # print(f"[Reservoir {self.net.index}]: first 5 predictions {predictions[0:5]}")
        
        return predictions












class DeepReservoir(torch.nn.Module):
    def __init__(self, config):
        """ Deep Reservoir module.
        The implementation realizes a number of stacked RNN modules using the
        ReservoirCell as core cell. All the reservoir modules share the same
        hyper-parameter values (i.e., same number of recurrent neurons, spectral
        radius, etc. ).

        :param config: dictionary containing parameters
        """
        super().__init__()
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.n_modules = config["n_modules"]
        self.mode = config["mode"]
        self.concat = config["concat"]
        self.batch_first = True  # DeepReservoir only supports batch_first
        self.tot_units = 0
        for i in range(self.n_modules):
            self.tot_units += config["reservoirs"][i]["units"]

        connectivity_input_1 = self.tot_units // self.n_modules
        connectivity_input_others = self.tot_units // self.n_modules
        connectivity_recurrent = self.tot_units // self.n_modules


        if self.concat:
            self.modules_units = int(self.tot_units / self.n_modules)
            # create the first reservoir:
            module = config["reservoirs"][0]
            reservoir_modules = [
                ReservoirModule(
                    input_size=module["input_size"],
                    output_size=module["output_size"],
                    units=module["units"],
                    index=0,
                    solver=module["solver"],
                    regul=module["regul"],
                    n_modules=config["n_modules"],
                    input_scaling=module["inp_scaling"],
                    spectral_radius=module["rho"],
                    leaky=module["leaky"],
                    connectivity_input=module["units"],
                    connectivity_recurrent=module["units"]//self.n_modules
                )
            ]
        else:
            self.modules_units = int(self.tot_units / self.n_modules)
            # create the first reservoir:
            module = config["reservoirs"][0]
            reservoir_modules = [
                ReservoirModule(
                    input_size=module["input_size"],
                    output_size=module["output_size"],
                    units=module["units"],
                    index=0,
                    solver=module["solver"],
                    regul=module["regul"],
                    n_modules=config["n_modules"],
                    input_scaling=module["inp_scaling"],
                    spectral_radius=module["rho"],
                    leaky=module["leaky"],
                    connectivity_input=module["units"],
                    connectivity_recurrent=module["units"]//self.n_modules
                )
            ]

        # create all the other reservoirs:
        for i in range(1, self.n_modules):
            module = config["reservoirs"][i]
            reservoir_modules.append(
                ReservoirModule(
                    input_size=module["input_size"],
                    output_size=module["output_size"],
                    units=module["units"],
                    index=i,
                    solver=module["solver"],
                    regul=module["regul"],
                    n_modules=config["n_modules"],
                    input_scaling=module["inp_scaling"],
                    spectral_radius=module["rho"],
                    leaky=module["leaky"],
                    connectivity_input=module["units"],
                    connectivity_recurrent=module["units"]//self.n_modules
                )
            )
            
        self.reservoirs = torch.nn.ModuleList(reservoir_modules)
        for res_module in self.reservoirs:
            res_module.net.verbose = False

    

    def forward(self, U, h_prev=None):
        """ compute the output of the deep reservoir.

        :param X:
        :return: hidden states (B, T, F), last state (L, B, F)
        """
        states = []  # list of all the states in all the modules
        states_last = []  # list of the states in all the modules for the last time step
        if h_prev is not None:
            states.append(h_prev)
        # states_last is a list because different modules may have different size.
        for res_idx, res_module in enumerate(self.reservoirs):
            # right now, i pass the input to each module, and also pass the states of the previous modules as neighbour values to the current module.

            [H, h_last] = res_module(U, h_prev=h_prev)
            
            states.append(H)
            states_last.append(h_last)

        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
        states_last = states_last[-1]
        return states, states_last
    

    def fit(self, U, Y, washout=200):
        """
        Function to train the deep reservoir's readout module (W_out matrix) given the dataset and the target.

        :param U: dataset
        :param Y: target
        :param washout: number of elements to remove from the beginning of the dataset
        :param solver: solver of the ridge regression
        :param regul: regularization coefficient
        """
        if self.n_modules > 1:
            # print("\n################## TRAINING ##################")
            if self.mode == "entangled":
                for m in range(self.n_modules):
                    # print(f"\n\n\n################## TRAINING MODULE {m} ##################")
                    U_module = U[:, :, m].reshape(1, -1, self.reservoirs[m].net.input_size)
                    Y_module = Y[:, (m+1)%Y.shape[1]].reshape(-1, self.reservoirs[m].net.output_size)
                    self.reservoirs[m].fit(U_module, Y_module, washout)
                # U_module = U[:, :, 2:3].reshape(1, -1, self.reservoirs[2].net.input_size)
                # Y_module = Y[:, 0].reshape(-1, self.reservoirs[2].net.output_size)
                # self.reservoirs[2].fit(U_module, Y_module, washout)
            elif self.mode == "independent":
                for m in range(self.n_modules):
                    U_module = U[:, :, m].reshape(1, -1, self.reservoirs[m].net.input_size)
                    Y_module = Y[:, m].reshape(-1, self.reservoirs[m].net.output_size)
                    self.reservoirs[m].fit(U_module, Y_module, washout)
            elif self.mode == "reinforced":
                for m in range(self.n_modules):
                    U_module = U[:, :, [m, ((m-1)%3)]].reshape(1, -1, self.reservoirs[m].net.input_size)
                    Y_module = Y[:, (m+1)%Y.shape[1]].reshape(-1, self.reservoirs[m].net.output_size)
                    self.reservoirs[m].fit(U_module, Y_module, washout)
            elif self.mode == "entangled_with_z":
                # print(f"\n\n\n################## TRAINING MODULE {m} ##################")
                U_module = U[:, :, 0].reshape(1, -1, 1)
                Y_module = Y[:, 1].reshape(-1, 1)
                self.reservoirs[0].fit(U_module, Y_module, washout)
                U_module = U[:, :, 1].reshape(1, -1, 1)
                Y_module = Y[:, 0].reshape(-1, 1)
                self.reservoirs[1].fit(U_module, Y_module, washout)
                U_module = U[:, :, 2].reshape(1, -1, 1)
                Y_module = Y[:, 2].reshape(-1, 1)
                self.reservoirs[2].fit(U_module, Y_module, washout)
            # print("################## TRAINING ##################\n")
        else:
            self.reservoirs[0].fit(U, Y, washout)



    def predict(self, n_iter, y_init=None, Y=None):
        """
        Function to predict the next n_iter timesteps

        :param n_iter: number of iterations to predict.
        :param y_init: possible initial values to start predicting from.
        """
        if self.n_modules > 1:
            if self.mode == "entangled":
                return self.predict_entangled(n_iter, y_init, Y)
            
            elif self.mode == "independent":
                predictions = [None for _ in range(self.n_modules)]
                for m in range(self.n_modules):
                    predictions[m] = self.reservoirs[m].predict(n_iter)
                    # print(f"Module [{m}] predictions: {predictions[m][0:10]}\n\n")
                predictions = torch.tensor(predictions, dtype=torch.float32).transpose(0, 1).reshape(-1, self.output_size)
                # print(f"\n\nFINAL PREDICTIONS: {predictions[0:10]}\n\n")
                return predictions
            
            elif self.mode == "reinforced":
                return self.predict_reinforced(n_iter, y_init, Y)
            
            elif self.mode == "entangled_with_z":
                return self.predict_entangled_with_z(n_iter, Y, y_init)
        else:
            return self.reservoirs[0].predict(n_iter)
    
    def predict_entangled(self, n_iter, y_init=None, Y=None):
        """
        Function to predict the next n_iter timesteps

        :param n_iter: number of iterations to predict.
        :param y_init: possible initial values to start predicting from.
        """
        if self.n_modules > 1:
            predictions = torch.tensor([], dtype=torch.float32)
            ot = [
                self.reservoirs[m].classifier.predict(
                    self.reservoirs[m].scaler.transform(
                        self.reservoirs[m].activations[-1].reshape(1, -1)
                    )
                )[0]
                for m in range(self.n_modules)
            ]
            ot = torch.tensor(ot, dtype=torch.float32).reshape(self.output_size, 1, 1, 1)
            ot = torch.cat([ot[-1:], ot[:-1]], dim=0)


            # ot[0] = torch.tensor(Y[0, 0], dtype=torch.float32).reshape(1, 1, 1)


            predictions = torch.cat([predictions, ot.reshape(1, self.output_size)], dim=0)
            past_prediction = ot
            ot = []
            for i in range(1, n_iter):
                for m in range(self.n_modules):
                    # module_input = past_prediction[m]
                    module_input = past_prediction[m].reshape(1, 1, 1)
                    new_activation = self.reservoirs[m](
                        module_input,
                        torch.tensor(self.reservoirs[m].activations[-1], dtype=torch.float32).reshape(1, -1)
                    )[0][0]
                    self.reservoirs[m].activations = np.concatenate(
                        [self.reservoirs[m].activations, new_activation.detach().numpy()], axis=0
                    )
                    ot.append(
                        self.reservoirs[m].classifier.predict(
                            self.reservoirs[m].scaler.transform(new_activation.detach().numpy().reshape(1, -1))
                        )[0]
                    )
                ot = torch.tensor(ot, dtype=torch.float32).reshape(self.output_size, 1, 1, 1)
                ot = torch.cat([ot[-1:], ot[:-1]], dim=0)

                # ot[0] = torch.tensor(Y[i, 0], dtype=torch.float32).reshape(1, 1, 1)


                predictions = torch.cat([predictions, ot.reshape(1, self.output_size)], dim=0)
                past_prediction = ot
                ot = []
            return torch.tensor(predictions, dtype=torch.float32)
        else:
            return self.reservoirs[0].predict(n_iter)
        

    def predict_entangled_with_z(self, n_iter, Y, y_init=None):
        """
        Function to predict the next n_iter timesteps

        :param n_iter: number of iterations to predict.
        :param y_init: possible initial values to start predicting from.
        """
        if self.n_modules > 1:
            predictions = torch.tensor([], dtype=torch.float32)
            ot = [
                self.reservoirs[m].classifier.predict(
                    self.reservoirs[m].scaler.transform(
                        self.reservoirs[m].activations[-1].reshape(1, -1)
                    )
                )[0]
                for m in range(self.n_modules)
            ]
            ot = torch.tensor(ot, dtype=torch.float32).reshape(3, 1, 1, 1)
            ot = torch.cat([ot[1], ot[0], ot[2]], dim=0).reshape(3, 1, 1, 1)

            # ot[0] = torch.tensor(Y[0, 0], dtype=torch.float32).reshape(1, 1, 1)


            predictions = torch.cat([predictions, ot.reshape(1, 3)], dim=0)
            past_prediction = ot
            ot = []
            for i in range(1, n_iter):
                for m in range(self.n_modules):
                    # module_input = past_prediction[m]
                    # print(Y[i, 2])
                    module_input = past_prediction[m]
                    new_activation = self.reservoirs[m](
                        module_input,
                        torch.tensor(self.reservoirs[m].activations[-1], dtype=torch.float32).reshape(1, -1)
                    )[0][0]
                    self.reservoirs[m].activations = np.concatenate(
                        [self.reservoirs[m].activations, new_activation.detach().numpy()], axis=0
                    )
                    ot.append(
                        self.reservoirs[m].classifier.predict(
                            self.reservoirs[m].scaler.transform(new_activation.detach().numpy().reshape(1, -1))
                        )[0]
                    )
                ot = torch.tensor(ot, dtype=torch.float32).reshape(3, 1, 1, 1)
                ot = torch.cat([ot[1], ot[0], ot[2]], dim=0).reshape(3, 1, 1, 1)

                # ot[1] = torch.tensor(Y[i, 1], dtype=torch.float32).reshape(1, 1, 1)


                predictions = torch.cat([predictions, ot.reshape(1, 3)], dim=0)
                past_prediction = ot
                ot = []
            return torch.tensor(predictions, dtype=torch.float32)
        else:
            return self.reservoirs[0].predict(n_iter)
    
    def predict_reinforced(self, n_iter, y_init=None, Y=None):
        """
        Function to predict the next n_iter timesteps

        :param n_iter: number of iterations to predict.
        :param y_init: possible initial values to start predicting from.
        """
        if self.n_modules > 1:
            # for m in range(self.n_modules):
            #     self.reservoirs[m].net.verbose = True
            # print("\n######################################################")
            # print(f"TIMESTEP {0}")
            # print("######################################################\n")
            predictions = torch.tensor([], dtype=torch.float32)
            ot = [
                self.reservoirs[m].classifier.predict(
                    self.reservoirs[m].scaler.transform(
                        self.reservoirs[m].activations[-1].reshape(1, -1)
                    )
                )[0]
                for m in range(self.n_modules)
            ]
            ot = torch.tensor(ot, dtype=torch.float32).reshape(3, 1, 1, 1)
            ot = torch.cat([ot[-1:], ot[:-1]], dim=0)


            # ot[0] = torch.tensor(Y[0, 0], dtype=torch.float32).reshape(1, 1, 1)


            predictions = torch.cat([predictions, ot.reshape(1, 3)], dim=0)
            past_prediction = ot
            ot = []
            for i in range(1, n_iter):
                for m in range(self.n_modules):
                    # module_input = past_prediction[m]
                    module_input = past_prediction[[m, ((m-1)%3)]].reshape(1, 1, self.reservoirs[m].net.input_size)
                    new_activation = self.reservoirs[m](
                        module_input,
                        torch.tensor(self.reservoirs[m].activations[-1], dtype=torch.float32).reshape(1, -1)
                    )[0][0]
                    self.reservoirs[m].activations = np.concatenate(
                        [self.reservoirs[m].activations, new_activation.detach().numpy()], axis=0
                    )
                    ot.append(
                        self.reservoirs[m].classifier.predict(
                            self.reservoirs[m].scaler.transform(new_activation.detach().numpy().reshape(1, -1))
                        )[0]
                    )
                # module_input = past_prediction[0:3].reshape(1, 1, 3)
                # new_activation = self.reservoirs[2](
                #     module_input, 
                #     torch.tensor(self.reservoirs[2].activations[-1], dtype=torch.float32).reshape(1, -1)
                # )[0][0]
                # self.reservoirs[2].activations = np.concatenate(
                #     [self.reservoirs[2].activations, new_activation.detach().numpy()], axis=0
                # )
                # ot.append(
                #     self.reservoirs[2].classifier.predict(
                #         self.reservoirs[2].scaler.transform(new_activation.detach().numpy().reshape(1, -1))
                #     )[0]
                # )
                ot = torch.tensor(ot, dtype=torch.float32).reshape(3, 1, 1, 1)
                ot = torch.cat([ot[-1:], ot[:-1]], dim=0)

                # ot[0] = torch.tensor(Y[i, 0], dtype=torch.float32).reshape(1, 1, 1)


                predictions = torch.cat([predictions, ot.reshape(1, 3)], dim=0)
                past_prediction = ot
                ot = []
            return torch.tensor(predictions, dtype=torch.float32)
        else:
            return self.reservoirs[0].predict(n_iter)