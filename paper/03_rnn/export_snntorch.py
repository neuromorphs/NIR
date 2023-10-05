import datetime

import nir

import torch
import torch.nn as nn
#from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import export
from snntorch import surrogate


export_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_name = "th1_bs5_collapsed"

export_path = "./Braille_snntorch_{}_{}".format(model_name,export_datetime)


def model_build(settings, input_size, num_steps, device):

    ### Network structure (input data --> encoding -> hidden -> output)
    input_channels = int(input_size)
    num_hidden = int(settings["nb_hidden"])
    num_outputs = 27

    ### Surrogate gradient setting
    spike_grad = surrogate.fast_sigmoid(slope=int(settings["slope"]))

    ### Put things together
    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            ##### Initialize layers #####
            self.fc1 = nn.Linear(input_channels, num_hidden)
            #self.lif1 = snn.RLeaky(beta=settings["beta_r"], linear_features=num_hidden, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif1 = snn.RSynaptic(alpha=settings["alpha_r"], beta=settings["beta_r"], linear_features=num_hidden, spike_grad=spike_grad, reset_mechanism="zero")
            ### Output layer
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            #self.lif2 = snn.Leaky(beta=settings["beta_out"], reset_mechanism="zero")
            self.lif2 = snn.Synaptic(alpha=settings["alpha_out"], beta=settings["beta_out"], spike_grad=spike_grad, reset_mechanism="zero")

        def forward(self, x):

            ##### Initialize hidden states at t=0 #####
            #spk1, mem1 = self.lif1.init_rleaky()
            spk1, syn1, mem1 = self.lif1.init_rsynaptic()
            #mem2 = self.lif2.init_leaky()
            syn2, mem2 = self.lif2.init_synaptic()

            # Record the spikes from the hidden layer (if needed)
            #spk1_rec = [] # not necessarily needed for inference
            # Record the final layer
            spk2_rec = []
            #syn2_rec = [] # not necessarily needed for inference
            #mem2_rec = [] # not necessarily needed for inference

            for step in range(num_steps):
                ### Recurrent layer
                cur1 = self.fc1(x[step])
                #spk1, mem1 = self.lif1(cur1, spk1, mem1)
                spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
                ### Output layer
                cur2 = self.fc2(spk1)
                #spk2, mem2 = self.lif2(cur2, mem2)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

                #spk1_rec.append(spk1) # not necessarily needed for inference
                spk2_rec.append(spk2)
                #syn2_rec.append(mem2) # not necessarily needed for inference
                #mem2_rec.append(mem2) # not necessarily needed for inference

            return torch.stack(spk2_rec, dim=0)

    return Net().to(device)


parameters = {
    "nb_hidden": 50,
    "alpha_r": 0.5, # just tentative
    "alpha_out": 0.2, # just tentative
    "beta_r": 0.85,
    "beta_out": 0.95,
    "lr": 0.005,
    "reg_l1": 0.0004,
    "reg_l2": 0.000007,
    "slope": 10
}

device = "cpu"

dummy_input = torch.rand(256, 1, 12)

saved_state_dict_path = "./model_ref_20230827_182756.pt"

model = model_build(settings=parameters, input_size=dummy_input.shape[-1], num_steps=dummy_input.shape[0], device=device)

model.load_state_dict(torch.load(saved_state_dict_path, map_location=device))

# Generate NIR graph
nir_graph = export.to_nir(model, dummy_input)

print("######################")
print("NIR graph exported!")
print("######################")

"""
torch.save(nir_graph, export_path)

print("####################")
print("NIR graph saved!")
print("####################")
"""

nir.write(export_path+".nir", nir_graph)

print("####################")
print("NIR graph written!")
print("####################")