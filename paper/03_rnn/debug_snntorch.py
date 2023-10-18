import json
import os
import numpy as np
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import nir
# NOTE: this requires snntorch/nir (PR) and nirtorch/master (unreleased)
from snntorch import import_nirtorch, export_nirtorch


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

device = torch.device("cpu")

saved_state_dict_path = "data/model_ref_20231011_100043.pt"
best_val_layers = torch.load(saved_state_dict_path, map_location=device)
parameters_path = "data/parameters_ref_20231011.json"
with open(parameters_path) as f:
    parameters = json.load(f)
regularization = [parameters["reg_l1"], parameters["reg_l2"]]

loss_fn = SF.ce_count_loss()

test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)

SHUFFLE = False

letter_written = ["Space", "A", "E", "I", "O", "U", "Y"]


def print_nir_graph(nir_graph: nir.NIRGraph):
    print("nodes:")
    for nodekey, node in nir_graph.nodes.items():
        print("\t", nodekey, node.__class__.__name__)
    print("edges:")
    for edge in nir_graph.edges:
        print("\t", edge)


def check_parameters(net, net2):
    print('\ncheck parameter match\n')
    w1 = net._modules['fc1']._parameters['weight']
    w2 = net2._modules['fc1']._parameters['weight']
    if not torch.allclose(w1, w2):
        print(f'input weights:     {torch.allclose(w1, w2)}')
    b1 = net._modules['fc1']._parameters['bias']
    b2 = net2._modules['fc1']._parameters['bias']
    if not torch.allclose(b1, b2):
        print(f'input bias:        {torch.allclose(b1, b2)}')
    w1 = net._modules['fc2']._parameters['weight']
    w2 = net2._modules['fc2']._parameters['weight']
    if not torch.allclose(w1, w2):
        print(f'output weights:    {torch.allclose(w1, w2)}')
    b1 = net._modules['fc2']._parameters['bias']
    b2 = net2._modules['fc2']._parameters['bias']
    if not torch.allclose(b1, b2):
        print(f'output bias:       {torch.allclose(b1, b2)}')
    w1 = net._modules['lif1'].recurrent._parameters['weight']
    w2 = net2._modules['lif1'].recurrent._parameters['weight']
    if not torch.allclose(w1, w2):
        print(f'recurrent weights: {torch.allclose(w1, w2)}')
    b1 = net._modules['lif1'].recurrent._parameters['bias']
    b2 = net2._modules['lif1'].recurrent._parameters['bias']
    if not torch.allclose(b1, b2):
        print(f'recurrent bias:    {torch.allclose(b1, b2)}')

    alpha1 = net._modules['lif1'].alpha
    alpha2 = net2._modules['lif1'].alpha
    if not torch.allclose(alpha1, alpha2):
        print(f'lif1 alpha:        {torch.allclose(alpha1, alpha2)}')
    beta1 = net._modules['lif1'].beta
    beta2 = net2._modules['lif1'].beta
    if not torch.allclose(beta1, beta2):
        print(f'lif1 beta:         {torch.allclose(beta1, beta2)}')
    alpha1 = net._modules['lif2'].alpha
    alpha2 = net2._modules['lif2'].alpha
    if not torch.allclose(alpha1, alpha2):
        print(f'lif2 alpha:        {torch.allclose(alpha1, alpha2)}')
    beta1 = net._modules['lif2'].beta
    beta2 = net2._modules['lif2'].beta
    if not torch.allclose(beta1, beta2):
        print(f'lif2 beta:         {torch.allclose(beta1, beta2)}')


def model_build(settings, input_size, num_steps, device):
    input_channels = int(input_size)
    num_hidden = int(settings["nb_hidden"])
    num_outputs = 7
    spike_grad = surrogate.fast_sigmoid(slope=int(settings["slope"]))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(input_channels, num_hidden)
            self.lif1 = snn.RSynaptic(
                alpha=settings["alpha_r"],
                beta=settings["beta_r"],
                linear_features=num_hidden,
                spike_grad=spike_grad,
                reset_mechanism="zero",
            )
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Synaptic(
                alpha=settings["alpha_out"],
                beta=settings["beta_out"],
                spike_grad=spike_grad,
                reset_mechanism="zero",
            )

        def forward(self, x):
            spk1, syn1, mem1 = self.lif1.init_rsynaptic()
            syn2, mem2 = self.lif2.init_synaptic()

            spk1_rec = []  # not necessarily needed for inference
            spk2_rec = []

            for step in range(num_steps):
                cur1 = self.fc1(x[step])
                spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

                spk1_rec.append(spk1)  # not necessarily needed for inference
                spk2_rec.append(spk2)

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

            loss_val = loss_fn(spk_out, labels)
            batch_loss.append(loss_val.detach().cpu().item())

            act_total_out = torch.sum(spk_out, 0)  # sum over time
            _, neuron_max_act_total_out = torch.max(act_total_out, 1)
            batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())

        return [np.mean(batch_loss), np.mean(batch_acc)]


def val_test_loop_nirtorch(
    dataset,
    batch_size,
    net,
    loss_fn,
    device,
    shuffle=True,
):
    with torch.no_grad():
        net.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        batch_loss = []
        batch_acc = []

        for data, labels in loader:
            data = data.to(device).swapaxes(1, 0)
            labels = labels.to(device)

            # reset network states
            for node in net.graph.node_list:
                if isinstance(node.elem, snn.RSynaptic):
                    node.elem.spk, node.elem.syn, node.elem.mem = node.elem.init_rsynaptic()
                elif isinstance(node.elem, snn.Synaptic):
                    node.elem.syn, node.elem.mem = node.elem.init_synaptic()

            # NET WANTS (B, N) (no time!)
            spk_out_arr, hid_rec_arr = [], []
            for t in range(data.shape[0]):
                spk_out, hid_rec = net(data[t, :, :])
                spk_out_arr.append(spk_out)
                hid_rec_arr.append(hid_rec)

            spk_out = torch.stack(spk_out_arr, dim=0)

            loss_val = loss_fn(spk_out, labels)
            batch_loss.append(loss_val.detach().cpu().item())

            act_total_out = torch.sum(spk_out, 0)  # sum over time
            _, neuron_max_act_total_out = torch.max(act_total_out, 1)
            batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())

        return [np.mean(batch_loss), np.mean(batch_acc)]


# build initial network
###########################

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
    shuffle=SHUFFLE,
    saved_state_dict=best_val_layers,
)
print("Test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))

# export to NIR
###########################

print('\nRNN graph with NIRTorch\n')
nir_graph = export_nirtorch.to_nir(net, ds_test[0][0])
# print_nir_graph(nir_graph)
nir.write("braille_v2.nir", nir_graph)

# import from NIR
###########################

net2 = import_nirtorch.from_nir(nir_graph)
check_parameters(net, net2)

# # HACK: remove self-recurrence of lif1  # DOES NOT FIX IT!
# # print(net2.graph.debug_str())
# lif1_node = {el.name: el for el in [e for e in net2.graph.node_list][-1].outgoing_nodes}['lif1']
# [e for e in net2.graph.node_list][-1].outgoing_nodes.pop(lif1_node)
# # print()
# # print(net2.graph.debug_str())
# # print()

# forward pass through both networks
###########################

loader = DataLoader(ds_test, batch_size=4, shuffle=SHUFFLE, drop_last=False)
data, labels = next(iter(loader))

# reset network 1 states
spk1, syn1, mem1 = net._modules['lif1'].init_rsynaptic()
syn2, mem2 = net._modules['lif2'].init_synaptic()

# reset network 2 states
for node in net2.graph.node_list:
    if isinstance(node.elem, snn.RSynaptic):
        node.elem.spk, node.elem.syn, node.elem.mem = node.elem.init_rsynaptic()
    elif isinstance(node.elem, snn.Synaptic):
        node.elem.syn, node.elem.mem = node.elem.init_synaptic()

sout1_arr, hrec1_arr = [], []
sout2_arr, hrec2_arr = [], []
h2_state = None
for tstep in range(data.shape[1]):
    x = data[:, tstep, :]

    # forward pass through network 1
    cur1 = net._modules['fc1'](x)
    spk1, syn1, mem1 = net._modules['lif1'](cur1, spk1, syn1, mem1)
    # Output layer
    cur2 = net._modules['fc2'](spk1)
    spk2, syn2, mem2 = net._modules['lif2'](cur2, syn2, mem2)
    sout1_arr.append(spk2)

    # forward pass through network 2
    spk_out, h2_state = net2(x, h2_state)
    sout2_arr.append(spk_out)

    # if not torch.equal(spk_out, spk2):
    #     print(tstep, spk_out.sum(), spk2.sum())


print('\n test the re-imported torch network\n')
test_results = val_test_loop_nirtorch(
    ds_test,
    batch_size,
    net2,
    loss_fn,
    device,
    shuffle=SHUFFLE,
)
print("Test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))

# back to NIR and test
###########################

net2 = import_nirtorch.from_nir(nir_graph)  # reset the network

print('\nback to NIR\n')
nir_graph2 = export_nirtorch.to_nir(net2, ds_test[0][0])
# print_nir_graph(nir_graph2)
nir.write("braille_v2.nir", nir_graph2)  # same, but without recurrent edge
nir_graph = export_nirtorch.to_nir(net, ds_test[0][0])  # must reload, bc graph was modified

assert nir_graph.nodes.keys() == nir_graph2.nodes.keys(), 'node keys mismatch'
for nodekey in nir_graph.nodes:
    a = nir_graph.nodes[nodekey].__class__.__name__ if nodekey in nir_graph.nodes else None
    b = nir_graph2.nodes[nodekey].__class__.__name__ if nodekey in nir_graph2.nodes else None
    assert a == b, f'node type mismatch: {a} vs {b}'
    # print(f'{nodekey}: {a}')
    for attr in nir_graph.nodes[nodekey].__dict__:
        close = None
        if isinstance(nir_graph.nodes[nodekey].__dict__[attr], np.ndarray):
            close = np.allclose(nir_graph.nodes[nodekey].__dict__[attr],
                                nir_graph2.nodes[nodekey].__dict__[attr])
        # print(f'\t{attr:12}: {close} {"!!!" if close is False else ""}')
        assert close is not False, f'node attribute mismatch: {nodekey}.{attr}'
