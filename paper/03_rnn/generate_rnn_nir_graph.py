import json
import os
import numpy as np
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# NIR stuff
import nir
# NOTE: this requires snntorch/nir (PR) and nirtorch/master (unreleased)
# from snntorch import export_nir
from snntorch import import_nirtorch, export_nirtorch


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# DEVICE SETTINGS
use_gpu = False

if use_gpu:
    gpu_sel = 1
    device = torch.device("cuda:" + str(gpu_sel))
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
else:
    device = torch.device("cpu")

# TRAINED WEIGHTS
saved_state_dict_path = "data/model_ref_20231011_100043.pt"
best_val_layers = torch.load(saved_state_dict_path, map_location=device)

# OPTIMAL HYPERPARAMETERS
parameters_path = "data/parameters_ref_20231011.json"

with open(parameters_path) as f:
    parameters = json.load(f)

regularization = [parameters["reg_l1"], parameters["reg_l2"]]

# LOSS FUNCTION
loss_fn = SF.ce_count_loss()
# TEST DATA

test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)

letter_written = ["Space", "A", "E", "I", "O", "U", "Y"]


def model_build(settings, input_size, num_steps, device):
    """Network structure (input data --> encoding -> hidden -> output)"""
    input_channels = int(input_size)
    num_hidden = int(settings["nb_hidden"])
    num_outputs = 7

    # Surrogate gradient setting
    spike_grad = surrogate.fast_sigmoid(slope=int(settings["slope"]))

    # Put things together
    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            # Initialize layers #
            self.fc1 = nn.Linear(input_channels, num_hidden)
            # self.lif1 = snn.RLeaky(beta=settings["beta_r"], linear_features=num_hidden,
            # spike_grad=spike_grad, reset_mechanism="zero")
            self.lif1 = snn.RSynaptic(
                alpha=settings["alpha_r"],
                beta=settings["beta_r"],
                linear_features=num_hidden,
                spike_grad=spike_grad,
                reset_mechanism="zero",
            )
            # Output layer
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            # self.lif2 = snn.Leaky(beta=settings["beta_out"], reset_mechanism="zero")
            self.lif2 = snn.Synaptic(
                alpha=settings["alpha_out"],
                beta=settings["beta_out"],
                spike_grad=spike_grad,
                reset_mechanism="zero",
            )

        def forward(self, x):
            # Initialize hidden states at t=0 #
            # spk1, mem1 = self.lif1.init_rleaky()
            spk1, syn1, mem1 = self.lif1.init_rsynaptic()
            # mem2 = self.lif2.init_leaky()
            syn2, mem2 = self.lif2.init_synaptic()

            # Record the spikes from the hidden layer (if needed)
            spk1_rec = []  # not necessarily needed for inference
            # Record the final layer
            spk2_rec = []
            # syn2_rec = [] # not necessarily needed for inference
            # mem2_rec = [] # not necessarily needed for inference

            for step in range(num_steps):
                # Recurrent layer
                cur1 = self.fc1(x[step])
                # spk1, mem1 = self.lif1(cur1, spk1, mem1)
                spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
                # Output layer
                cur2 = self.fc2(spk1)
                # spk2, mem2 = self.lif2(cur2, mem2)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

                spk1_rec.append(spk1)  # not necessarily needed for inference
                spk2_rec.append(spk2)
                # syn2_rec.append(mem2) # not necessarily needed for inference
                # mem2_rec.append(mem2) # not necessarily needed for inference

            return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0)

    return Net().to(device)


def val_test_loop(
    dataset,
    batch_size,
    net,
    loss_fn,
    device,
    shuffle=True,
    saved_state_dict=None,
    label_probabilities=False,
    regularization=None,
):
    with torch.no_grad():
        if saved_state_dict is not None:
            net.load_state_dict(saved_state_dict)
        net.eval()

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
        )

        batch_loss = []
        batch_acc = []

        for data, labels in loader:
            data = data.to(device).swapaxes(1, 0)
            labels = labels.to(device)

            spk_out, hid_rec = net(data)

            # Validation loss
            if regularization is not None:
                # L1 loss on spikes per neuron from the hidden layer
                reg_loss = regularization[0] * torch.mean(torch.sum(hid_rec, 0))
                # L2 loss on total number of spikes from the hidden layer
                reg_loss = reg_loss + regularization[1] * torch.mean(
                    torch.sum(torch.sum(hid_rec, dim=0), dim=1) ** 2
                )
                loss_val = loss_fn(spk_out, labels) + reg_loss
            else:
                loss_val = loss_fn(spk_out, labels)

            batch_loss.append(loss_val.detach().cpu().item())

            # Accuracy
            act_total_out = torch.sum(spk_out, 0)  # sum over time
            _, neuron_max_act_total_out = torch.max(
                act_total_out, 1
            )  # argmax over output units to compare to labels
            batch_acc.extend(
                (neuron_max_act_total_out == labels).detach().cpu().numpy()
            )
            # batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy

        if label_probabilities:
            log_softmax_fn = nn.LogSoftmax(dim=-1)
            log_p_y = log_softmax_fn(act_total_out)
            return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y)
        else:
            return [np.mean(batch_loss), np.mean(batch_acc)]


def val_test_loop_nirtorch(
    dataset,
    batch_size,
    net,
    loss_fn,
    device,
    shuffle=True,
    label_probabilities=False,
):
    with torch.no_grad():
        net.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        batch_loss = []
        batch_acc = []

        for data, labels in loader:
            data = data.to(device).swapaxes(1, 0)
            labels = labels.to(device)

            print('data.shape', data.shape)
            print('labels.shape', labels.shape)

            # TODO: implement the forward pass correctly (iterate over time)

            # TODO: reset the state of the network
            for node in net.graph.node_list:
                if isinstance(node.elem, snn.RSynaptic):
                    node.elem.spk, node.elem.syn, node.elem.mem = node.elem.init_rsynaptic()
                elif isinstance(node.elem, snn.Synaptic):
                    node.elem.syn, node.elem.mem = node.elem.init_synaptic()
                elif isinstance(node.elem, snn.RLeaky):
                    node.elem.spk, node.elem.mem = node.elem.init_rleaky()
                elif isinstance(node.elem, snn.Leaky):
                    node.elem.mem = node.elem.init_leaky()

            # NET WANTS (B, N) (no time!)
            spk_out_arr, hid_rec_arr = [], []
            for t in range(data.shape[0]):
                spk_out, hid_rec = net(data[t, :, :])
                spk_out_arr.append(spk_out)
                hid_rec_arr.append(hid_rec)

            spk_out = torch.stack(spk_out_arr, dim=0)

            print('spk_out.shape', spk_out.shape)
            print()

            # Validation loss
            loss_val = loss_fn(spk_out, labels)
            batch_loss.append(loss_val.detach().cpu().item())

            # Accuracy
            act_total_out = torch.sum(spk_out, 0)  # sum over time
            _, neuron_max_act_total_out = torch.max(
                act_total_out, 1
            )  # argmax over output units to compare to labels
            batch_acc.extend(
                (neuron_max_act_total_out == labels).detach().cpu().numpy()
            )
            # batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy

        if label_probabilities:
            log_softmax_fn = nn.LogSoftmax(dim=-1)
            log_p_y = log_softmax_fn(act_total_out)
            return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y)
        else:
            return [np.mean(batch_loss), np.mean(batch_acc)]


# INFERENCE ON TEST SET

batch_size = 64
input_size = 12
num_steps = next(iter(ds_test))[0].shape[0]
net = model_build(parameters, input_size, num_steps, device)
test_results = val_test_loop(
    ds_test,
    batch_size,
    net,
    loss_fn,
    device,
    shuffle=False,
    saved_state_dict=best_val_layers,
    regularization=regularization,
)
print("Test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))

# # INFERENCE ON INDIVIDUAL TEST SAMPLES

# Ns = 10

# for ii in range(Ns):
#     single_sample = next(iter(DataLoader(ds_test, batch_size=1, shuffle=True)))
#     _, lbl_probs = val_test_loop(
#         TensorDataset(single_sample[0], single_sample[1]),
#         1,
#         net,
#         loss_fn,
#         device,
#         shuffle=False,
#         saved_state_dict=best_val_layers,
#         label_probabilities=True,
#         regularization=regularization,
#     )
#     print("Single-sample inference {}/{} from test set:".format(ii + 1, Ns))
#     print(
#         "Sample: {} \tPrediction: {}".format(
#             letter_written[single_sample[1]],
#             letter_written[torch.max(lbl_probs.cpu(), 1)[1]],
#         )
#     )
#     print(
#         "Label probabilities (%): {}\n".format(
#             np.round(np.array(lbl_probs.detach().cpu().numpy()) * 100, 2)
#         )
#     )


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

def print_nir_graph(nir_graph: nir.NIRGraph):
    print("nodes:")
    for nodekey, node in nir_graph.nodes.items():
        print("\t", nodekey, node.__class__.__name__)
    print("edges:")
    for edge in nir_graph.edges:
        print("\t", edge)


# nir_graph = export_nir.to_nir(net, ds_test[0][0])
# print('\nRNN graph without NIRTorch\n')
# print_nir_graph(nir_graph)
# nir.write("braille.nir", nir_graph)

nir_graph = export_nirtorch.to_nir(net, ds_test[0][0])
print('\nRNN graph with NIRTorch\n')
# print_nir_graph(nir_graph)
nir.write("braille_v2.nir", nir_graph)

net2 = import_nirtorch.from_nir(nir_graph)

# check that parameters are the same in both networks
print('\ncheck parameter match\n')
w1 = net._modules['fc1']._parameters['weight']
w2 = net2._modules['fc1']._parameters['weight']
print(f'input weights:     {torch.allclose(w1, w2)}')
b1 = net._modules['fc1']._parameters['bias']
b2 = net2._modules['fc1']._parameters['bias']
print(f'input bias:        {torch.allclose(b1, b2)}')
w1 = net._modules['fc2']._parameters['weight']
w2 = net2._modules['fc2']._parameters['weight']
print(f'output weights:    {torch.allclose(w1, w2)}')
b1 = net._modules['fc2']._parameters['bias']
b2 = net2._modules['fc2']._parameters['bias']
print(f'output bias:       {torch.allclose(b1, b2)}')
w1 = net._modules['lif1'].recurrent._parameters['weight']
w2 = net2._modules['lif1'].recurrent._parameters['weight']
print(f'recurrent weights: {torch.allclose(w1, w2)}')
b1 = net._modules['lif1'].recurrent._parameters['bias']
b2 = net2._modules['lif1'].recurrent._parameters['bias']
print(f'recurrent bias:    {torch.allclose(b1, b2)}')

alpha1 = net._modules['lif1'].alpha
alpha2 = net2._modules['lif1'].alpha
print(f'lif1 alpha:        {alpha1 == alpha2}')
beta1 = net._modules['lif1'].beta
beta2 = net2._modules['lif1'].beta
print(f'lif1 beta:         {beta1 == beta2}')
alpha1 = net._modules['lif2'].alpha
alpha2 = net2._modules['lif2'].alpha
print(f'lif2 alpha:        {alpha1 == alpha2}')
beta1 = net._modules['lif2'].beta
beta2 = net2._modules['lif2'].beta
print(f'lif2 beta:         {beta1 == beta2}')

loader = DataLoader(ds_test, batch_size=64, shuffle=True, drop_last=False)
data, labels = next(iter(loader))

for node in net2.graph.node_list:
    if isinstance(node.elem, snn.RSynaptic):
        node.elem.spk, node.elem.syn, node.elem.mem = node.elem.init_rsynaptic()
    elif isinstance(node.elem, snn.Synaptic):
        node.elem.syn, node.elem.mem = node.elem.init_synaptic()
    elif isinstance(node.elem, snn.RLeaky):
        node.elem.spk, node.elem.mem = node.elem.init_rleaky()
    elif isinstance(node.elem, snn.Leaky):
        node.elem.mem = node.elem.init_leaky()


# HACK: remove self-recurrence of lif1
# [e for e in net2.graph.node_list][-1].outgoing_nodes.pop({el.name: el for el in [e for e in net2.graph.node_list][-1].outgoing_nodes}['lif1'])

print('\n test the re-imported torch network\n')
batch_size = 64
input_size = 12
num_steps = next(iter(ds_test))[0].shape[0]
# net = model_build(parameters, input_size, num_steps, device)
test_results = val_test_loop_nirtorch(
    ds_test,
    batch_size,
    net2,
    loss_fn,
    device,
    shuffle=False,
)
print("Test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))

net2 = import_nirtorch.from_nir(nir_graph)  # reset the network

print('\nback to NIR\n')
nir_graph2 = export_nirtorch.to_nir(net2, ds_test[0][0])
# print_nir_graph(nir_graph2)
nir.write("braille_v2.nir", nir_graph2)  # same, but without recurrent edge

# important: reload original nir_graph bc it was modified
nir_graph = export_nirtorch.to_nir(net, ds_test[0][0])

assert nir_graph.nodes.keys() == nir_graph2.nodes.keys(), 'node keys mismatch'

for nodekey in nir_graph.nodes:
    a = nir_graph.nodes[nodekey].__class__.__name__ if nodekey in nir_graph.nodes else None
    b = nir_graph2.nodes[nodekey].__class__.__name__ if nodekey in nir_graph2.nodes else None
    assert a == b, f'node type mismatch: {a} vs {b}'
    print(f'{nodekey}: {a}')
    for attr in nir_graph.nodes[nodekey].__dict__:
        close = None
        if isinstance(nir_graph.nodes[nodekey].__dict__[attr], np.ndarray):
            close = np.allclose(nir_graph.nodes[nodekey].__dict__[attr],
                                nir_graph2.nodes[nodekey].__dict__[attr])
        print(f'\t{attr:12}: {close} {"!!!" if close is False else ""}')
