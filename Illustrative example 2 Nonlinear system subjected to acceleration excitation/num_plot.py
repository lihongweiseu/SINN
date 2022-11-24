# %% This .py code is used to plot results
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

params = {"text.usetex": True, "font.family": "serif", "font.serif": ["Times"],
          'text.latex.preamble': [r'\usepackage{fontspec}'
                                  r'\usepackage{newtxtext,newtxmath} \usepackage{amsmath} \usepackage{amssymb}']
          }
plt.rcParams.update(params)

# Prepare data
dt = 0.05
Nt = 1001
tend = 50
t = np.linspace(0, tend, Nt).reshape(-1, 1)
NTrain = 10  # number of ground motions for training
NTest = 89  # number of ground motions for testing
f = loadmat('num_data_ref.mat')
Train_y_ref = np.transpose(f['Train_y_ref'])
Test_y_ref = np.transpose(f['Test_y_ref'])

f = loadmat('num_data_pred0.mat')
Train_y_pred0 = np.transpose(f['Train_y_pred0'])
Test_y_pred0 = np.transpose(f['Test_y_pred0'])

f = loadmat('num_data_pred1_2layer1_2state_2neuron1_3000.mat')
Train_y_pred1 = np.transpose(f['Train_y_pred1'])
Test_y_pred1 = np.transpose(f['Test_y_pred1'])
del f

# Calculate the Pearson correlation coefficients
Train_corr0 = np.zeros((NTrain, 1))
Train_corr1 = np.zeros((NTrain, 1))
Test_corr0 = np.zeros((NTest, 1))
Test_corr1 = np.zeros((NTest, 1))

for i in range(NTrain):
    A = Train_y_ref[:, i:i + 1].reshape(1, -1)
    B = Train_y_pred0[:, i:i + 1].reshape(1, -1)
    Train_corr0[i:i + 1, :] = np.corrcoef(A, B)[0, 1]
    B1 = Train_y_pred1[:, i:i + 1].reshape(1, -1)
    Train_corr1[i:i + 1, :] = np.corrcoef(A, B1)[0, 1]

for i in range(NTest):
    A = Test_y_ref[:, i:i + 1].reshape(1, -1)
    B = Test_y_pred0[:, i:i + 1].reshape(1, -1)
    Test_corr0[i:i + 1, :] = np.corrcoef(A, B)[0, 1]
    B = Test_y_pred1[:, i:i + 1].reshape(1, -1)
    Test_corr1[i:i + 1, :] = np.corrcoef(A, B)[0, 1]

corr0 = np.vstack((Train_corr0, Test_corr0))
corr1 = np.vstack((Train_corr1, Test_corr1))
cm = 1 / 2.54
N = np.linspace(1, NTrain + NTest, NTrain + NTest).reshape(-1, 1)

# %% Plot the Pearson correlation coefficients over datasets
fig = plt.figure(figsize=(16 * cm, 8 * cm))
ax = fig.add_subplot(111)
ax.plot([NTrain+0.5, NTrain+0.5], [0.5, 1], 'k', lw=1)
ax.scatter(N, corr0, c='b', marker='o', s=50, alpha=0.8, linewidths=0, label='PhyCNN')
ax.scatter(N, corr1, c='r', marker='^', s=50, alpha=0.8, linewidths=0, label='SINN')
ax.set_ylim([0.5, 1])
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.tick_params(axis='y', labelsize=8)
ax.set_ylabel(r'Pearson corr.', fontsize=8, labelpad=1)
ax.set_xlim([0, 100])
ax.set_xticks(np.arange(0, 100.1, 10.0))
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel(r'Case no.', fontsize=8, labelpad=1)
ax.text(5, 0.55, 'Training', ha='center', va='center', fontsize=8)
ax.text(35, 0.55, 'Testing', ha='center', va='center', fontsize=8)
legend = ax.legend(loc='lower right', bbox_to_anchor=(0.875, 0.03), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                   edgecolor='black', fontsize=8, ncol=1, columnspacing=0.5, handletextpad=0.0)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legendHandles:
    obj.set_lw(0.75)
ax.tick_params(direction="in")  # plt.
ax.grid(lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.1)
plt.show()
# fig.savefig(fname = "F_num_corr.pdf", format = "pdf")

# %% Plot the time histories for datasets 29 30 65 94
col = ['k', 'b', 'r']
c_dash = [[8, 4], [2, 2]]
fig = plt.figure(figsize=(16 * cm, 12 * cm))
ax = fig.add_subplot(411)
ax.plot(t, Test_y_ref[:, 18:19], color=col[0], lw=0.5, label='Ground truth')
ax.plot(t, Test_y_pred0[:, 18:19], color=col[1], dashes=[8, 4], lw=0.5, label='PhyCNN')
ax.plot(t, Test_y_pred1[:, 18:19], color=col[2], dashes=[2, 2], lw=0.5, label='SINN')
ax.set_ylim([-0.6, 0.6])
ax.set_yticks(np.arange(-0.6, 0.61, 0.3))
ax.text(40, -0.45, 'Case 29', ha='center', va='center', fontsize=8)
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                   edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legendHandles:
    obj.set_lw(0.75)

ax = fig.add_subplot(412)
ax.plot(t, Test_y_ref[:, 19:20], color=col[0], lw=0.5)
ax.plot(t, Test_y_pred0[:, 19:20], color=col[1], dashes=c_dash[0], lw=0.5)
ax.plot(t, Test_y_pred1[:, 19:20], color=col[2], dashes=c_dash[1], lw=0.5)
ax.set_ylim([-0.4, 0.4])
ax.set_yticks(np.arange(-0.4, 0.41, 0.2))
ax.text(40, -0.3, 'Case 30', ha='center', va='center', fontsize=8)

ax = fig.add_subplot(413)
ax.plot(t, Test_y_ref[:, 54:55], color=col[0], lw=0.5)
ax.plot(t, Test_y_pred0[:, 54:55], color=col[1], dashes=[8, 4], lw=0.5)
ax.plot(t, Test_y_pred1[:, 54:55], color=col[2], dashes=[2, 2], lw=0.5)
ax.set_ylim([-0.2, 0.2])
ax.set_yticks(np.arange(-0.2, 0.21, 0.1))
ax.text(40, -0.15, 'Case 65', ha='center', va='center', fontsize=8)

ax = fig.add_subplot(414)
ax.plot(t, Test_y_ref[:, 83:84], color=col[0], lw=0.5)
ax.plot(t, Test_y_pred0[:, 83:84], color=col[1], dashes=c_dash[0], lw=0.5)
ax.plot(t, Test_y_pred1[:, 83:84], color=col[2], dashes=c_dash[1], lw=0.5)
ax.set_ylim([-0.2, 0.2])
ax.set_yticks(np.arange(-0.2, 0.21, 0.1))
ax.set_xlabel(r'Time (s)', fontsize=8, labelpad=1)
ax.text(40, -0.15, 'Case 94', ha='center', va='center', fontsize=8)

for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    ax.tick_params(axis = 'y', labelsize=8)
    ax.set_xlim([0, 50])
    ax.set_xticks(np.arange(0, 50.1, 5))
    ax.tick_params(axis = 'x', labelsize=8)
    ax.set_ylabel(r'Disp. (m)', fontsize=8, labelpad = 1)
    ax.tick_params(direction="in")  # plt.
    plt.grid(lw=0.5)
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0.25, wspace=0.15)
plt.show()
# fig.savefig(fname = "F_num_dis.pdf", format = "pdf")
