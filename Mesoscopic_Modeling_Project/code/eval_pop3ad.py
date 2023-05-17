import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(self):
        super(MultiTaskNet, self).__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=30, num_layers=1, batch_first=True)
        self.fc_J_syn = nn.Linear(30, 3*3)
        self.fc_mu = nn.Linear(30, 3)
        self.fc_tau_m = nn.Linear(30, 3)
        self.fc_V_th = nn.Linear(30, 3)
        self.fc_J_theta = nn.Linear(30, 3)
        self.fc_tau_theta = nn.Linear(30, 3)

    def forward(self, x):
        x, _ = self.gru(x)  # Shape of output x: (batch_size, seq_len, hidden_size)
                            # Shape of output x: (batch_size, seq_len, input_size)
        x = x[:, -1, :]  # We only need the last time step's output

        J_syn = self.fc_J_syn(x).view(-1, 3, 3)
        mu = self.fc_mu(x)
        tau_m = self.fc_tau_m(x)
        V_th = self.fc_V_th(x)
        J_theta = self.fc_J_theta(x).view(-1, 3, 1)
        tau_theta = self.fc_tau_theta(x).view(-1, 3, 1)

        return J_syn, mu, tau_m, V_th, J_theta, tau_theta

class MultiTaskNet_big(nn.Module):
    def __init__(self):
        super(MultiTaskNet_big, self).__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=100, num_layers=1, batch_first=True)
        self.fc_J_syn = nn.Linear(100, 3*3)
        self.fc_mu = nn.Linear(100, 3)
        self.fc_tau_m = nn.Linear(100, 3)
        self.fc_V_th = nn.Linear(100, 3)
        self.fc_J_theta = nn.Linear(100, 3)
        self.fc_tau_theta = nn.Linear(100, 3)

    def forward(self, x):
        x, _ = self.gru(x)  # Shape of output x: (batch_size, seq_len, hidden_size)
                            # Shape of output x: (batch_size, seq_len, input_size)
        x = x[:, -1, :]  # We only need the last time step's output

        J_syn = self.fc_J_syn(x).view(-1, 3, 3)
        mu = self.fc_mu(x)
        tau_m = self.fc_tau_m(x)
        V_th = self.fc_V_th(x)
        J_theta = self.fc_J_theta(x).view(-1, 3, 1)
        tau_theta = self.fc_tau_theta(x).view(-1, 3, 1)

        return J_syn, mu, tau_m, V_th, J_theta, tau_theta

def test_model(sample_filename, model_filename, max_values_A_N, max_values_labels):
    # Load the model
    # model = MultiTaskNet()
    model = MultiTaskNet_big()
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the sample data
    with open(sample_filename, 'r') as json_file:
        data_dict = json.loads(json_file.read())

    A_N = np.array(data_dict["A_N"])
    A_N_reshaped = A_N.reshape(-1, 120, 3)  # Reshape to (60000, 3) -> (500, 120, 3)
    A_N_avg = A_N_reshaped.mean(axis=1)  # Average over the second dimension, resulting in a (500, 3) array

    # Normalize A_N_avg to [-1, 1] for each channel
    A_N_avg = A_N_avg / max_values_A_N[np.newaxis, :] if np.any(max_values_A_N > 0) else A_N_avg

    # Convert A_N_avg to tensor
    A_N_avg = torch.from_numpy(A_N_avg).float()
    A_N_avg = A_N_avg.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Compute the model output
    with torch.no_grad():
        outputs = model(A_N_avg)

    # Denormalize the outputs and convert them to numpy arrays
    outputs_denorm = [output.squeeze(0).cpu().numpy() * max_val for output, max_val in zip(outputs, max_values_labels)]

    # Load the ground truth labels
    labels_raw = (
        np.array(data_dict["J_syn"]),
        np.array(data_dict["mu"]),
        np.array(data_dict["tau_m"]),
        np.array(data_dict["V_th"]),
        np.array(data_dict["J_theta"]),
        np.array(data_dict["tau_theta"])
    )
    labels = [l for l in labels_raw]

    return outputs_denorm, labels


# Give the max values as reference (which are not necessarily max values in data)
max_values_A_N = np.array([0.14360417, 0.13154167, 0.149875]) 
max_values_labels = np.array([2.10217639, 59.73844498, 39.97235326, 29.6845174,
                              1487.77125213, 1491.84213703])

# Assume your json files are in the "data_folder"
data_folder = 'pop3_data_with_adapt_test'
filenames = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder) if fn.endswith('.json')]
filenames = random.sample(filenames, 20)

# Evaluate and compare
path_to_sample_file = filenames[0]
path_to_saved_model = "./trained_models/model_pop3_best_v2.pth"
outputs, labels = test_model(path_to_sample_file, path_to_saved_model, max_values_A_N, max_values_labels)
print("output:\n", outputs)
print("labels:\n", labels)
# calculate MSE loss with normalization
outputs = [torch.from_numpy(l / max_val).float() if max_val > 0 else torch.from_numpy(l) 
                    for l, max_val in zip(outputs, max_values_labels)]
labels = [torch.from_numpy(l / max_val).float() if max_val > 0 else torch.from_numpy(l) 
                    for l, max_val in zip(labels, max_values_labels)]
loss = sum(F.mse_loss(output, label) for output, label in zip(outputs, labels))
print("MSE loss:\n", loss)