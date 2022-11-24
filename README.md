# Codes for the paper "State-integration neural network for modelling of forced-vibration systems"
---  coded in the Python environment utilizing the powerful deep learning library PyTorch.
# # Illustrative example 1 Shear Zener model of viscoelastic dampers
1. Zener_Sin_RNN_main.py is the main Python file for the quick denmonstration of the RNN model.
2. Zener_BLWN_SINN_main.py is the main Python file for the SINN model.
3. Zener_BLWN_preparation.m is the matlab file to generate the banded limited white noises as input signals.
4. Zener_BLWN_data.mat is the generated banded limited white noises.
5. Zener_BLWN_SINN_2layer1_2state_2neuron1_6000_10s_loss2834em1.pt is the trained library of SINN model, users could load it in Zener_BLWN_SINN_main.py to genetate Zener_BLWN_SINN_pred_2layer1_2state_2neuron1_6000_10s.mat quickly.
6. Zener_BLWN_SINN_1layer_2state_2neuron1_3000_10s_loss1739ep0_fail.pt is the trained library of SINN model (failed case), users could load it in Zener_BLWN_SINN_main.py to genetate Zener_BLWN_SINN_pred_1layer_2state_2neuron1_3000_10s_fail.mat quickly. To match this trained library, "state_layer_non = 2" should be changed to "state_layer_non = 1" in Zener_BLWN_SINN_main.py
7. Zener_BLWN_SINN_pred_2layer1_2state_2neuron1_6000_10s.mat stores the simualtion results for Fig.7 in the paper.
8. Zener_BLWN_SINN_pred_1layer_2state_2neuron1_3000_10s_fail.mat stores the simualtion results for Fig.8 (failed case) in the paper.
9. Zener_BLWN_SINN_plot.py plots the results.
