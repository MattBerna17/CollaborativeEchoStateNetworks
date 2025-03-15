from sklearn.linear_model import Ridge
import torch
import numpy as np
from utils import get_lorenz_attractor, plot_lorenz_attractor_with_error
from esn_alternative import sparse_eye_init, sparse_recurrent_tensor_init, sparse_tensor_init, spectral_norm_scaling

class Reservoir(torch.nn.Module):
    def __init__(self, Nu, Nx, Ny, input_scaling=1., leaky=1, spectral_radius=0.9, W_connectivity_perc=0.1, use_bias=False, use_feedback=False):
        """
        Initialize the reservoir with random weights
        :param Nu: input size
        :param Nx: reservoir size
        :param Ny: output size
        :param use_bias: whether to use bias
        :param use_feedback: whether to use feedback
        """
        super().__init__()
        self.Nu = Nu
        self.Nx = Nx
        self.Ny = Ny
        self.use_bias = use_bias
        self.use_feedback = use_feedback
        self.input_scaling = input_scaling
        self.leaky = leaky
        self.spectral_radius = spectral_radius
        self.W_connectivity_perc = W_connectivity_perc
        self.W_in = torch.nn.Parameter(sparse_tensor_init(self.Nx, self.Nu, self.Nu) * self.input_scaling, requires_grad=False) # dimension: Nx x Nu
        W = sparse_recurrent_tensor_init(self.Nx, C=int(self.Nx*self.W_connectivity_perc))
        if self.leaky == 1:
            W = spectral_norm_scaling(W, self.spectral_radius)
        else:
            I = sparse_eye_init(self.Nx)
            W = W * self.leaky + (I * (1 - self.leaky))
            W = spectral_norm_scaling(W, self.spectral_radius)
            self.W = (W + I * (self.leaky - 1)) * (1 / self.leaky)
        self.W = torch.nn.Parameter(W, requires_grad=False) # dimension: Nx x Nx
        self.W_out = torch.nn.Parameter(sparse_tensor_init(self.Ny, self.Nx, self.Nx)) # dimension: Ny x Nx ---> output does not depend directly on the bias and input, only on the reservoir state
        if use_bias:
            self.b = torch.nn.Parameter(sparse_tensor_init(3, self.Nx, self.Nx), requires_grad=False) # bias dimension: Nx
        if use_feedback:
            self.W_fb = torch.nn.Parameter(sparse_tensor_init(self.Nx, self.Ny, self.Ny), requires_grad=False) # dimension: Nx x Ny
        self.y_prev = torch.zeros(Ny) # previous output which is initially only zeros
        self.x_prev = torch.zeros(Nx) # previous state which is initially only zeros
        self.X = [] # store the states of the reservoir
        self.Y = [] # store the outputs of the reservoir
        self.U = [] # store the inputs of the reservoir
        self.ridge = Ridge(alpha=0.05, fit_intercept=False)
        # print(f'W_in: {self.W_in.shape}')
        # print(f'W: {self.W.shape}')
        # print(f'W_out: {self.W_out.shape}')

    def forward(self, u, y_prev=None): # teacher forcing
        """
        Forward pass of the reservoir
        :param u: input
        :param x: reservoir state
        :return: output
        """
        self.U = np.append(self.U, u) # append current input to history of inputs
        self.x_prev = torch.tanh(
            self.W_in @ u +                                         # input weights
            self.W @ self.x_prev +                                  # reservoir weights
            (self.W_fb @ y_prev if self.use_feedback and y_prev is not None else 0)   # feedback weights
            # (self.b if self.use_bias else 0)                        # bias #!TODO: fix bias dimension
        )
        # print(f'x_prev: {self.x_prev.shape}')
        self.X.append(self.x_prev) # append current state to history of states
        # !TODO: change readout position: it should not be in the forward, since it needs X+!
        self.y_prev = self.W_out @ self.x_prev  # compute output for the current state (not considering directly the input and the bias)
        #TODO: add leaky term
        self.Y.append(self.y_prev) # append current output to history of outputs
        return self.y_prev                      # return prevision
    
    def fit(self, U, Y_target, learning_rate=0.01, epochs=1, use_backprop=False):
        """
        Fit the reservoir to the input data
        :param U: input data
        :param Y_target: target data
        """
        assert U.shape[0] == Y_target.shape[0]
        if (use_backprop):
            criterion = torch.nn.MSELoss() # the loss criterion is the Mean Squared Error
            optimizer = torch.optim.SGD([self.W_out], lr=learning_rate) # the optimizer is Stochastic Gradient Descent
            for epoch in range(epochs):
                for i in range(U.shape[0]):
                    y_pred = self.forward(U[i], Y_target[i-1])
                    # print(f'U[{i}]: {U[i].shape}')
                    # print(f'Y_pred: {y_pred.shape}')
                    loss = criterion(y_pred, Y_target[i])
                    optimizer.zero_grad() # reset the gradients to zero
                    loss.backward() # backpropagate the loss to compute the gradients
                    optimizer.step() # update the weights
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{U.shape[0]}], Loss: {loss.item():.4f}')
        else: 
            X_states = []
            for i in range(U.shape[0]):
                u = U[i]
                if (i == 0):
                    self.forward(u)
                else:
                    self.forward(u, Y_target[i-1])
                X_states.append(self.x_prev.detach().numpy())
            X_states = np.array(X_states)
            with open("weights_lorenz_alt.txt", "w") as f:
                for i in range(X_states.shape[0]):
                    for j in range(X_states.shape[1]):
                        f.write(str(X_states[i][j]) + ',')
                    f.write('\n')
            ridge = Ridge(alpha=0.05, fit_intercept=False)
            # print(f'X_states: {X_states.shape}')
            # print(f'Y_target: {Y_target.shape}')
            ridge.fit(X_states, Y_target)
            self.W_out.data = torch.tensor(ridge.coef_, dtype=torch.float32)
    
    def load_W(self, file):
        with open(file, "r") as f:
            weights = []
            while True:
                line = f.readline()
                if not line:
                    break
                data = line.strip().split(',')[:-1]
                data = list(map(float, data))
                weights.append(data)
            # print(f'W: {weights}')
            weights = np.array(weights).reshape(self.Nx, self.Nx)
            self.W = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

    def load_W_in(self, file):
        with open(file, "r") as f:
            weights = []
            while True:
                line = f.readline()
                if not line:
                    break
                data = line.strip().split(',')[:-1]
                data = list(map(float, data))
                weights.append(data)
            # print(f'W_in: {weights}')
            weights = np.array(weights).reshape(self.Nx, self.Nu)
            self.W_in = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

    def load_W_out(self, file):
        with open(file, "r") as f:
            weights = []
            while True:
                line = f.readline()
                if not line:
                    break
                data = line.strip().split(',')[:-1]
                data = list(map(float, data))
                weights.append(data)
            # print(f'W_out: {weights}')
            weights = np.array(weights).reshape(self.Ny, self.Nx)
            self.W_out = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

    def predict(self, U):
        """
        Predict the output of the reservoir
        :param U: input data
        :return: predicted output
        """
        Y_pred = torch.Tensor()
        pred = None
        for i in range(U.shape[0]):
            pred = self.forward(U[i])
            print(f'Prediction shape: {pred.shape}')
            print(f'Prediction: {pred}')
            Y_pred = torch.cat((Y_pred, pred))
        return Y_pred.reshape(-1, self.Ny).detach().numpy()




if __name__ == "__main__":
    (train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_lorenz_attractor(washout=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NRMSEs = []
    for i in range(1):
        model = Reservoir(Nu=3, Nx=512, Ny=3, spectral_radius=0.7, leaky=0.4, input_scaling=1, use_feedback=False).to(device)
        # model.fit(train_dataset, train_target, epochs=1, use_backprop=False)
        print(f'W_in: {model.W_in.shape}')
        print(f'W: {model.W.shape}')
        print(f'W_out: {model.W_out.shape}')
        print("\n\n")
        model.load_W_in("W_in.txt")
        model.load_W("W.txt")
        model.load_W_out("W_out.txt")
        print(f'W_in: {model.W_in.shape}')
        print(f'W: {model.W.shape}')
        print(f'W_out: {model.W_out.shape}')
        model.ridge.coef_ = model.W_out.detach().numpy()
        if model.use_feedback:
            print(f'W_fb: {model.W_fb.shape}')
        Y_pred = model.predict(test_dataset)
        # print(f'Predictions: {Y_pred.shape}')
        print(f'Predictions: {Y_pred[:10]}')
        print(f'Target: {test_target[:10]}')
        plot_lorenz_attractor_with_error(torch.Tensor(Y_pred), test_target, 'Lorenz attractor')
        mse = np.mean(np.square(Y_pred - test_target.numpy()))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(test_target).mean())
        nrmse = rmse / (norm + 1e-9)
        NRMSEs.append(nrmse)
    nrmse = np.mean(NRMSEs)
    print(f'NRMSE: {nrmse:.4f}')