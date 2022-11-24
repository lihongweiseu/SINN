# %% This code is to test SINN by using Zener model with BLWN inputs
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import math
import random
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from scipy.io import savemat, loadmat

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
state_size = 2  # user defined

state_layer_non = 2  # number of nonlinear layers for state derivatives # state_layer_non = 1
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
f = loadmat('Zener_BLWN_data.mat')
Train_dt = f['Train_dt'][0, 0]
Train_Nt = f['Train_Nt'][0, 0]
Train_data = f['Train_data'].item()  # t u y

Test_dt = f['Test_dt'][:, 0]
Test_Nt = f['Test_Nt'][:, 0]
Test_data = f['Test_data'][:, 0]
# # Test_data[0].item() includes datasets 2~5 (t u y)
# # Test_data[1].item() includes datasets 6~9 (t u y)
Test_data = (Test_data[0].item(), Test_data[1].item())
del f

Train_u = torch.tensor(Train_data[1], dtype=torch.float)
Train_y_ref = torch.tensor(Train_data[2], dtype=torch.float).to(device)
# %% Training
optimizer = optim.Adam(SINN_model.parameters(), 0.01)
training_num = 6000 # 3000
Train_tend = 10  # the first Train_tend seconds of data are used for training
Train_Nt0 = math.floor(Train_tend / Train_dt) + 1
loss_all = np.zeros((training_num + 1, 1))  # store loss values
start = time.time()
for counter in range(training_num + 1):
    loss = torch.tensor([[0]], dtype=torch.float).to(device)
    x = torch.zeros(1, state_size)
    for i in range(Train_Nt0 - 1):
        u = Train_u[:, i:i + 1]
        x0 = x
        u_x = torch.cat((u, x), dim=1)
        y = SINN_model(u_x)[:, 0:output_size].to(device)
        loss += criterion(y, Train_y_ref[:, i:i + 1])
        k1 = SINN_model(u_x)[:, output_size:]

        u = (u + Train_u[:, i + 1:i + 2]) / 2
        x = x0 + k1 * Train_dt / 2
        u_x = torch.cat((u, x), dim=1)
        k2 = SINN_model(u_x)[:, output_size:]

        x = x0 + k2 * Train_dt / 2
        u_x = torch.cat((u, x), dim=1)
        k3 = SINN_model(u_x)[:, output_size:]

        u = Train_u[:, i + 1:i + 2]
        x = x0 + k3 * Train_dt
        u_x = torch.cat((u, x), dim=1)
        k4 = SINN_model(u_x)[:, output_size:]

        x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * Train_dt / 6

    u_x = torch.cat((u, x), dim=1)
    y = SINN_model(u_x)[:, 0:output_size].to(device)
    loss += criterion(y, Train_y_ref[:, i + 1:i + 2])
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
# torch.save(SINN_model.state_dict(), "Zener_BLWN_SINN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt")
# torch.save(SINN_model.state_dict(), "Zener_BLWN_SINN_1layer_2state_2neuron1_3000_10s_loss1739ep0_fail.pt")

# # Load trained model if the users do not want to train the model again
# SINN_model.load_state_dict(torch.load("Zener_BLWN_SINN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt"))
# SINN_model.load_state_dict(torch.load("Zener_BLWN_SINN_1layer_2state_2neuron1_3000_10s_loss1739ep0_fail.pt"))


# %% Plot training result
Train_y_pred = np.zeros((1, Train_Nt))
x = torch.zeros(1, state_size)
for i in range(Train_Nt - 1):
    u = Train_u[:, i:i + 1]
    x0 = x
    u_x = torch.cat((u, x), dim=1)
    y = SINN_model(u_x)[:, 0:output_size]
    Train_y_pred[:, i:i + 1] = y.detach().numpy()
    k1 = SINN_model(u_x)[:, output_size:]

    u = (u + Train_u[:, i + 1:i + 2]) / 2
    x = x0 + k1 * Train_dt / 2
    u_x = torch.cat((u, x), dim=1)
    k2 = SINN_model(u_x)[:, output_size:]

    x = x0 + k2 * Train_dt / 2
    u_x = torch.cat((u, x), dim=1)
    k3 = SINN_model(u_x)[:, output_size:]

    u = Train_u[:, i + 1:i + 2]
    x = x0 + k3 * Train_dt
    u_x = torch.cat((u, x), dim=1)
    k4 = SINN_model(u_x)[:, output_size:]

    x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * Train_dt / 6

u_x = torch.cat((u, x), dim=1)
y = SINN_model(u_x)[:, 0:output_size]
Train_y_pred[:, Train_Nt - 1:Train_Nt] = y.detach().numpy()
plt.plot(Train_data[0].reshape(-1, 1), Train_y_ref.reshape(-1, 1))
plt.plot(Train_data[0].reshape(-1, 1), Train_y_pred.reshape(-1, 1))
plt.show()

# %% Plot testing result
Test_y_pred = (np.zeros((4, Test_Nt[0])), np.zeros((4, Test_Nt[1])))
x = torch.zeros(4, state_size)
for j in range(2):
    Test_u = torch.tensor(Test_data[j][1], dtype=torch.float)
    for i in range(Test_Nt[j] - 1):
        u = Test_u[:, i:i + 1]
        x0 = x
        u_x = torch.cat((u, x), dim=1)
        y = SINN_model(u_x)[:, 0:output_size]
        Test_y_pred[j][:, i:i + 1] = y.detach().numpy()
        k1 = SINN_model(u_x)[:, output_size:]

        u = (u + Test_u[:, i + 1:i + 2]) / 2
        x = x0 + k1 * Test_dt[j] / 2
        u_x = torch.cat((u, x), dim=1)
        k2 = SINN_model(u_x)[:, output_size:]

        x = x0 + k2 * Test_dt[j] / 2
        u_x = torch.cat((u, x), dim=1)
        k3 = SINN_model(u_x)[:, output_size:]

        u = Test_u[:, i + 1:i + 2]
        x = x0 + k3 * Test_dt[j]
        u_x = torch.cat((u, x), dim=1)
        k4 = SINN_model(u_x)[:, output_size:]

        x = x0 + (k1 + 2 * k2 + 2 * k3 + k4) * Test_dt[j] / 6

    u_x = torch.cat((u, x), dim=1)
    y = SINN_model(u_x)[:, 0:output_size]
    Test_y_pred[j][:, Test_Nt[j] - 1:Test_Nt[j]] = y.detach().numpy()

j = 1
i = 0
plt.plot(Test_data[j][0][0:1, :].reshape(-1, 1), Test_data[j][2][i:i+1, :].reshape(-1, 1))
plt.plot(Test_data[j][0][0:1, :].reshape(-1, 1), Test_y_pred[j][i:i+1, :].reshape(-1, 1))
plt.show()

# %% save data
# savemat('Zener_BLWN_SINN_pred_2layer1_2state_2neuron1_6000_10s.mat',
        # {'Train_y_pred': Train_y_pred, 'Test_y_pred_1_4': Test_y_pred[0],'Test_y_pred_5_8': Test_y_pred[1]})

# savemat('Zener_BLWN_SINN_pred_1layer_2state_2neuron1_3000_10s_fail.mat',
#         {'Train_y_pred': Train_y_pred, 'Test_y_pred_1_4': Test_y_pred[0],'Test_y_pred_5_8': Test_y_pred[1]})
