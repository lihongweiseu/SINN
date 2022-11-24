# %% This .py code is used to plot results
# author: Hong-Wei Li, Email: hongweili@seu.edu.cn

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy.io import loadmat

params = {"text.usetex": True, "font.family": "serif", "font.serif": ["Times"],
          'text.latex.preamble': [r'\usepackage{fontspec}'
                                  r'\usepackage{newtxtext,newtxmath} \usepackage{amsmath} \usepackage{amssymb}']
          }
plt.rcParams.update(params)
cm = 1 / 2.54

# Prepare data
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

f = loadmat('Zener_BLWN_SINN_pred_1layer_2state_2neuron1_3000_10s_fail.mat')
Train_data = Train_data + (f['Train_y_pred'],)  # t u y y_pred
temp1 = Test_data[0] + (f['Test_y_pred_1_4'],)
temp2 = Test_data[1] + (f['Test_y_pred_5_8'],)
Test_data = (temp1, temp2)  # t u y y_pred
del f

# Calculate the Pearson correlation coefficients
Train_Test_corr = np.zeros((9, 1))
Train_Test_corr[0] = np.corrcoef(Train_data[2], Train_data[3])[0, 1]
for i in range(2):
    for j in range(4):
        k = 4 * i + j + 1
        Train_Test_corr[k] = np.corrcoef(Test_data[i][2][j, :], Test_data[i][3][j, :])[0, 1]

print("------Pearson correlation coefficients------")
print("Case 1: %.8f" % Train_Test_corr[0])
for i in range(8):
    print("Case %d: %.8f" % (i + 2, Train_Test_corr[i + 1]))

# %% Plot results for Case 4
col = ['b', 'r']
c_dash = [[8, 4], [9, 4, 2, 4], [2, 2]]
tend = 50
Test_Nt0 = math.floor(tend / Test_dt[0]) + 1
fig2 = plt.figure(figsize = (16 * cm, 6 * cm))
ax = fig2.add_subplot(111)
ax.plot(Test_data[0][0][0,  0:Test_Nt0], Test_data[0][2][2, 0:Test_Nt0], color = col[0], dashes = c_dash[0], lw = 0.5,
        label = 'Ground truth')
ax.plot(Test_data[0][0][0,  0:Test_Nt0], Test_data[0][3][2, 0:Test_Nt0], color = col[1], dashes = c_dash[1], lw = 0.5,
        label = 'Prediction')
ax.set_ylabel(r'Force. (kN)', fontsize = 8, labelpad = 1)
ax.set_xlim([0, 50])
ax.set_xticks(np.arange(0, 50.1, 5))
ax.set_xlabel(r'Time (s)', fontsize = 8, labelpad = 1)
ax.set_ylim([-4, 8])
ax.set_yticks(np.arange(-4, 8.1, 2))
ax.tick_params(axis = 'y', labelsize = 8)
ax.tick_params(axis = 'x', labelsize = 8)
ax.tick_params(direction = "in")  # plt.
ax.grid(lw = 0.5)
ax.set_axisbelow(True)

text_box = AnchoredText(r'Case 4', frameon = True, loc = 'upper center', pad = 0.2,
                        borderpad = 0.8, prop = dict(fontsize = 8))
plt.setp(text_box.patch, facecolor = 'white', alpha = 1, lw = .75)
ax.add_artist(text_box)
legend = ax.legend(loc = 'upper right', bbox_to_anchor = (0.95, 0.95), borderpad = 0.2, borderaxespad = 0,
                   handlelength = 2.8,
                   edgecolor = 'black', fontsize = 8, ncol = 1, columnspacing = 0.5,
                   handletextpad = 0.3)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad = 0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for legobj in legend.legendHandles:
    legobj.set_lw(0.75)
ax.tick_params(direction = "in")  # plt.
ax.grid(lw = 0.5)


fig2.tight_layout(pad = 0.1)
# plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
plt.show()
# fig2.savefig(fname = "F_Zener_BLWN_case4_fail.pdf", format = "pdf")
