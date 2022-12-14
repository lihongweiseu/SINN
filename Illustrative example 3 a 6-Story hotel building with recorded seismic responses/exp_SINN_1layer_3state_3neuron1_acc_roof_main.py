# %% State integration neural network (SINN) for the experimental case (Sec 3.1)  of Ref:
# Physics-guided convolutional neural network (PhyCNN) for data-driven seismic response modeling
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat, loadmat
from torch import nn, optim

# chose to use gpu or cpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# To guarantee same results for every running, which might slow down the training speed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

# Define the configuration of SINN
input_size = 1
output_size = 1
state_size = 3  # user defined

state_layer_non = 1  # number of nonlinear layers for state derivatives
state_neuron_non = np.zeros(state_layer_non, dtype=np.int32)  # size of each nonlinear layer
for i in range(state_layer_non):
    state_neuron_non[i] = state_size  # user defined

output_layer_non = 1  # number of nonlinear layers for outputs
output_neuron_non = np.zeros(output_layer_non, dtype=np.int32)  # size of each nonlinear layer
for i in range(output_layer_non):
    output_neuron_non[i] = output_size  # user defined


# Define SINN
class SINN(nn.Module):
    def __init__(self):
        super(SINN, self).__init__()

        layer_non = [nn.Linear(input_size + state_size, state_neuron_non[0], bias=False), torch.nn.Tanh()]
        if state_layer_non > 1:
            for ii in range(state_layer_non - 1):
                layer_non.append(nn.Linear(state_neuron_non[ii], state_neuron_non[ii + 1], bias=False))
                layer_non.append(torch.nn.Tanh())
        layer_non.append(nn.Linear(state_neuron_non[-1], state_size, bias=False))
        self.StateNet_non = nn.Sequential(*layer_non)
        self.StateNet_lin = nn.Linear(input_size + state_size, state_size, bias=False)

        layer_non = [nn.Linear(input_size + state_size, output_neuron_non[0], bias=False), torch.nn.Tanh()]
        if output_layer_non > 1:
            for ii in range(output_layer_non - 1):
                layer_non.append(nn.Linear(output_neuron_non[ii], output_neuron_non[ii + 1], bias=False))
                layer_non.append(torch.nn.Tanh())
        layer_non.append(nn.Linear(output_neuron_non[-1], output_size, bias=False))
        self.OutputNet_non = nn.Sequential(*layer_non)
        self.OutputNet_lin = nn.Linear(input_size + state_size, output_size, bias=False)

    def forward(self, input_state):
        state_d_non = self.StateNet_non(input_state)
        state_d_lin = self.StateNet_lin(input_state)
        state_d = state_d_non + state_d_lin
        output_non = self.OutputNet_non(input_state)
        output_lin = self.OutputNet_lin(input_state)
        output = output_non + output_lin
        output_state_d = torch.cat((output, state_d), dim=1)
        return output_state_d


# Create model
SINN_model = SINN()
criterion = nn.MSELoss()

# Prepare data
dt = 0.02
Nt = 2500
tend = 49.98
t = np.linspace(0, tend, Nt).reshape(-1, 1)
NTrain = 15  # number of ground motions for training
NTest = 6  # number of ground motions for testing
f = loadmat('results_exp_ag2utt.mat')
Train_u = f['X_train'].reshape(NTrain, Nt)
Train_u_torch = torch.tensor(Train_u, dtype=torch.float)
Train_y_ref = f['ytt_train_ref'][0:NTrain, :, 1:2].reshape(NTrain, Nt)
Train_y_ref_torch = torch.tensor(Train_y_ref, dtype=torch.float).to(device)
Train_y_pred0 = f['ytt_train_pred'][0:NTrain, :, 1:2].reshape(NTrain, Nt)

loss0 = f['train_loss'].reshape(-1, 1)
Test_u = f['X_pred'][0:NTest, :, :].reshape(NTest, Nt)
Test_u_torch = torch.tensor(Test_u, dtype=torch.float)
Test_y_ref = f['ytt_pred_ref'][0:NTest, :, 1:2].reshape(NTest, Nt)
Test_y_pred0 = f['ytt_pred'][0:NTest, :, 1:2].reshape(NTest, Nt)

# %% Training
optimizer = optim.Adam(SINN_model.parameters(), 0.01)
training_num = 10000
loss_all = np.zeros((training_num + 1, 1))  # store loss values
start = time.time()
for counter in range(training_num + 1):
    loss = torch.tensor([[0]], dtype=torch.float).to(device)
    x = torch.zeros(NTrain, state_size)
    for i in range(Nt - 1):  # 1001 data points for each ground motion
        u = Train_u_torch[:, i:i + 1]
        x0 = x
        u_x = torch.cat((u, x), dim=1)
        y = SINN_model(u_x)[:, 0:output_size].to(device)
        loss += criterion(y, Train_y_ref_torch[:, i:i + 1])
        k1 = SINN_model(u_x)[:, output_size:]

        u = (u + Train_u_torch[:, i + 1:i + 2]) / 2
        x = x0 + k1 * dt / 2
        u_x = torch.cat((u, x), dim=1)
        k2 = SINN_model(u_x)[:, output_size:]

        x = x0 + k2 * dt / 2
        u_x = torch.cat((u, x), dim=1)
        k3 = SINN_model(u_x)[:, output_size:]

        u = Train_u_torch[:, i + 1:i + 2]
        x = x0 + k3 * dt
        u_x = torch.cat((u, x), dim=1)
        k4 = SINN_model(u_x)[:, output_size:]

        x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    u_x = torch.cat((u, x), dim=1)
    y = SINN_model(u_x)[:, 0:output_size].to(device)
    loss += criterion(y, Train_y_ref_torch[:, i + 1:i + 2])
    loss_all[counter:counter + 1, :] = loss.item()
    if counter < training_num:
        SINN_model.zero_grad()
        loss.backward()
        optimizer.step()

    if (counter + 1) % 10 == 0 or counter == 0:
        print(f"iteration: {counter + 1}, loss: {loss.item()}")
        end = time.time()
        per_time = (end - start) / (counter + 1)
        print("Average training time: %.3f s per training" % per_time)
        print("Cumulative training time: %.3f s" % (end - start))
        left_time = (training_num - counter + 1) * per_time
        print(f"Executed at {time.strftime('%H:%M:%S', time.localtime())},", "left time: %.3f s\n" % left_time)

end = time.time()
print("Total training time: %.3f s" % (end - start))
print(f"loss: {loss.item()}")
# # Save model
# torch.save(SINN_model.state_dict(), "exp_SINN_1layer_3state_3neuron1_acc_roof_10000_loss9413ep3.pt")

# # Load trained model if the users do not want to train the model again
# SINN_model.load_state_dict(torch.load("exp_SINN_1layer_3state_3neuron1_acc_roof_10000_loss9413ep3.pt"))

# %% Plot training result
Train_y_pred1 = np.zeros((NTrain, Nt))
x = torch.zeros(NTrain, state_size)
for i in range(Nt - 1):  # 1001 data points for each ground motion
    u = Train_u_torch[:, i:i + 1]
    x0 = x
    u_x = torch.cat((u, x), dim=1)
    y = SINN_model(u_x)[:, 0:output_size]
    Train_y_pred1[:, i:i + 1] = y.detach().numpy()
    k1 = SINN_model(u_x)[:, output_size:]

    u = (u + Train_u_torch[:, i + 1:i + 2]) / 2
    x = x0 + k1 * dt / 2
    u_x = torch.cat((u, x), dim=1)
    k2 = SINN_model(u_x)[:, output_size:]

    x = x0 + k2 * dt / 2
    u_x = torch.cat((u, x), dim=1)
    k3 = SINN_model(u_x)[:, output_size:]

    u = Train_u_torch[:, i + 1:i + 2]
    x = x0 + k3 * dt
    u_x = torch.cat((u, x), dim=1)
    k4 = SINN_model(u_x)[:, output_size:]

    x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

u_x = torch.cat((u, x), dim=1)
y = SINN_model(u_x)[:, 0:output_size]
Train_y_pred1[:, Nt - 1:Nt] = y.detach().numpy()
i = 9  # select a random integer between 0 and NTrain to check the training performance
plt.plot(t, Train_y_ref[i:i + 1, :].reshape(-1, 1))
plt.plot(t, Train_y_pred1[i:i + 1, :].reshape(-1, 1))
plt.show()

# %% Plot testing result
Test_y_pred1 = np.zeros((NTest, Nt))
x = torch.zeros(NTest, state_size)
for i in range(Nt - 1):  # 1001 data points for each ground motion
    u = Test_u_torch[:, i:i + 1]
    x0 = x
    u_x = torch.cat((u, x), dim=1)
    y = SINN_model(u_x)[:, 0:output_size]
    Test_y_pred1[:, i:i + 1] = y.detach().numpy()
    k1 = SINN_model(u_x)[:, output_size:]

    u = (u + Test_u_torch[:, i + 1:i + 2]) / 2
    x = x0 + k1 * dt / 2
    u_x = torch.cat((u, x), dim=1)
    k2 = SINN_model(u_x)[:, output_size:]

    x = x0 + k2 * dt / 2
    u_x = torch.cat((u, x), dim=1)
    k3 = SINN_model(u_x)[:, output_size:]

    u = Test_u_torch[:, i + 1:i + 2]
    x = x0 + k3 * dt
    u_x = torch.cat((u, x), dim=1)
    k4 = SINN_model(u_x)[:, output_size:]

    x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

u_x = torch.cat((u, x), dim=1)
y = SINN_model(u_x)[:, 0:output_size]
Test_y_pred1[:, Nt - 1:Nt] = y.detach().numpy()
i = 2  # select a random integer between 0 and NTest to check the testing performance
plt.plot(t, Test_y_ref[i:i + 1, :].reshape(-1, 1))
plt.plot(t, Test_y_pred1[i:i + 1, :].reshape(-1, 1))
plt.show()

# %% save data
# savemat('exp_data_ref_acc_roof.mat', {'Train_u': Train_u, 'Train_y_ref': Train_y_ref,
#                              'Test_u': Test_u, 'Test_y_ref': Test_y_ref})
# savemat('exp_data_pred0_acc_roof.mat', {'Train_y_pred0': Train_y_pred0, 'Test_y_pred0': Test_y_pred0})
# savemat('exp_data_pred1_1layer_3state_3neuron1_acc_roof_10000.mat', {'Train_y_pred1': Train_y_pred1, 'Test_y_pred1': Test_y_pred1})