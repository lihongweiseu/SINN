# Codes for the paper "Modelling of forced-vibration systems using state-integration neural network"
---  coded in the Python environment utilizing the powerful deep learning library PyTorch.
## Illustrative example 1 Shear Zener model of viscoelastic dampers
1. _Zener_Sin_RNN_main.py_ is the main Python file for the quick denmonstration of the RNN model.
2. _Zener_BLWN_SINN_main.py_ is the main Python file for the SINN model.
3. _Zener_BLWN_preparation.m_ is the matlab file to generate the banded limited white noises as input signals.
4. _Zener_BLWN_data.mat_ is the generated banded limited white noises.
5. _Zener_BLWN_SINN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt_ is the trained library of SINN model, users could load it in _Zener_BLWN_SINN_main.py_ to genetate _Zener_BLWN_SINN_pred_2layer1_2state_2neuron1_6000_10s.mat_ quickly.
6. _Zener_BLWN_SINN_1layer_2state_2neuron1_3000_10s_loss1739ep0_fail.pt_ is the trained library of SINN model (failed case), users could load it in _Zener_BLWN_SINN_main.py_ to genetate _Zener_BLWN_SINN_pred_1layer_2state_2neuron1_3000_10s_fail.mat_ quickly. To match this trained library, "state_layer_non = 2" should be changed to "state_layer_non = 1" in _Zener_BLWN_SINN_main.py_
7. _Zener_BLWN_SINN_pred_2layer1_2state_2neuron1_6000_10s.mat_ stores the simualtion results for Fig.7 in the paper.
8. _Zener_BLWN_SINN_pred_1layer_2state_2neuron1_3000_10s_fail.mat_ stores the simualtion results for Fig.8 (failed case) in the paper.
9. _Zener_BLWN_SINN_plot.py_ plots the results.

## Illustrative example 2 Nonlinear system subjected to acceleration excitation
1. _num_SINN_2layer1_2state_2neuron1_main.py_ is the main Python file for the SINN model.
2. _num_SINN_2layer1_2state_2neuron1_3000_loss5316em2.pt_ is the trained library of SINN model, users could load it in _num_SINN_2layer1_2state_2neuron1_main.py_ to genetate _num_data_pred1_2layer1_2state_2neuron1_3000.mat_ quickly.
3. _num_data_preparation.py_ prepares the data.
4. _results_num_ag2u.mat_ is the data file coming from the PhyCNN paper "Physics-guided convolutional neural network (PhyCNN) for data-driven seismic response modeling".
5. _num_data_ref.mat_ stores the reference results.
6. _num_data_pred0.mat_ stores the prediction results using the PhyCNN model.
7. _num_data_pred1_2layer1_2state_2neuron1_3000.mat_ stores the prediction results using the SINN model.
8. _num_plot.py_ plots the results.

## Illustrative example 3 a 6-Story hotel building with recorded seismic responses
1. _exp_SINN_1layer_3state_3neuron1_acc_roof_main.py_ is the main Python file for the SINN model.
2. _exp_SINN_1layer_3state_3neuron1_acc_roof_10000_loss9413ep3.pt_ is the trained library of SINN model, users could load it in _exp_SINN_1layer_3state_3neuron1_acc_roof_main.py_ to genetate _exp_data_pred1_1layer_3state_3neuron1_acc_roof_10000.mat_ quickly.
3. _results_exp_ag2utt.mat_ is the data file coming from the PhyCNN paper
4. _exp_data_ref_acc_roof.mat_ stores the reference results.
5. _exp_data_pred0_acc_roof.mat_ stores the prediction results using the PhyCNN model.
7. _exp_data_pred1_1layer_3state_3neuron1_acc_roof_10000.mat_ stores the prediction results using the SINN model.
8. _exp_plot.py_ plots the results.
