import json
import os
import numpy as np
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nir

# NOTE: this requires snntorch/nir (PR) and nirtorch/master (unreleased)
from snntorch import import_nirtorch, export_nirtorch
import nirtorch


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


def check_parameters(net, net2) -> bool:
    ok = True
    w1 = net._modules["fc1"]._parameters["weight"]
    w2 = net2._modules["fc1"]._parameters["weight"]
    if not torch.allclose(w1, w2):
        print(f"input weights:     {torch.allclose(w1, w2)}")
        ok = False
    b1 = net._modules["fc1"]._parameters["bias"]
    b2 = net2._modules["fc1"]._parameters["bias"]
    if not torch.allclose(b1, b2):
        print(f"input bias:        {torch.allclose(b1, b2)}")
        ok = False
    w1 = net._modules["fc2"]._parameters["weight"]
    w2 = net2._modules["fc2"]._parameters["weight"]
    if not torch.allclose(w1, w2):
        print(f"output weights:    {torch.allclose(w1, w2)}")
        ok = False
    b1 = net._modules["fc2"]._parameters["bias"]
    b2 = net2._modules["fc2"]._parameters["bias"]
    if not torch.allclose(b1, b2):
        print(f"output bias:       {torch.allclose(b1, b2)}")
        ok = False
    w1 = net._modules["lif1"].recurrent._parameters["weight"]
    w2 = net2._modules["lif1"].recurrent._parameters["weight"]
    if not torch.allclose(w1, w2):
        print(f"recurrent weights: {torch.allclose(w1, w2)}")
        ok = False
    b1 = net._modules["lif1"].recurrent._parameters["bias"]
    b2 = net2._modules["lif1"].recurrent._parameters["bias"]
    if not torch.allclose(b1, b2):
        print(f"recurrent bias:    {torch.allclose(b1, b2)}")
        ok = False

    alpha1 = net._modules["lif1"].alpha
    alpha2 = net2._modules["lif1"].alpha
    if not torch.allclose(alpha1, alpha2):
        print(f"lif1 alpha:        {torch.allclose(alpha1, alpha2)}")
        ok = False
    beta1 = net._modules["lif1"].beta
    beta2 = net2._modules["lif1"].beta
    if not torch.allclose(beta1, beta2):
        print(f"lif1 beta:         {torch.allclose(beta1, beta2)}")
        ok = False
    alpha1 = net._modules["lif2"].alpha
    alpha2 = net2._modules["lif2"].alpha
    if not torch.allclose(alpha1, alpha2):
        print(f"lif2 alpha:        {torch.allclose(alpha1, alpha2)}")
        ok = False
    beta1 = net._modules["lif2"].beta
    beta2 = net2._modules["lif2"].beta
    if not torch.allclose(beta1, beta2):
        print(f"lif2 beta:         {torch.allclose(beta1, beta2)}")
        ok = False
    return ok


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

            spk_out, _ = net(data)

            loss_val = loss_fn(spk_out, labels)
            batch_loss.append(loss_val.detach().cpu().item())

            act_total_out = torch.sum(spk_out, 0)  # sum over time
            _, neuron_max_act_total_out = torch.max(act_total_out, 1)
            batch_acc.extend(
                (neuron_max_act_total_out == labels).detach().cpu().numpy()
            )

        return [np.mean(batch_loss), np.mean(batch_acc)]


def val_test_loop_nirtorch(dataset, batch_size, net, loss_fn, device, shuffle=True):
    with torch.no_grad():
        net.eval()
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
        )

        batch_loss = []
        batch_acc = []

        for data, labels in loader:
            data = data.to(device).swapaxes(1, 0)
            labels = labels.to(device)

            h_state = nirtorch.from_nir.GraphExecutorState(
                state={
                    "lif1": net._modules[
                        "lif1"
                    ].init_rsynaptic(),  # 3-tuple: spk, syn, mem
                    "lif2": net._modules["lif2"].init_synaptic(),  # 2-tuple: syn, mem
                }
            )

            spk_out_arr = []
            for t in range(data.shape[0]):
                spk_out, h_state = net(data[t], h_state)
                spk_out_arr.append(spk_out)
            spk_out = torch.stack(spk_out_arr, dim=0)

            loss_val = loss_fn(spk_out, labels)
            batch_loss.append(loss_val.detach().cpu().item())

            act_total_out = torch.sum(spk_out, 0)  # sum over time
            _, neuron_max_act_total_out = torch.max(act_total_out, 1)
            batch_acc.extend(
                (neuron_max_act_total_out == labels).detach().cpu().numpy()
            )

        return [np.mean(batch_loss), np.mean(batch_acc)]


class ImportedNetwork(nn.Module):
    def __init__(self, nir_graph: nir.NIRGraph):
        super().__init__()
        self.graph = nir_graph

        node = nir_graph.nodes["fc1"]
        self.fc1 = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
        self.fc1.weight.data = torch.Tensor(node.weight)
        self.fc1.bias.data = torch.Tensor(node.bias)

        nodelif = nir_graph.nodes["lif1.lif"]
        nodewrec = nir_graph.nodes["lif1.w_rec"]
        self.lif1 = snn.RSynaptic(
            alpha=float(np.unique(1 - (1e-4 / nodelif.tau_syn))[0]),
            beta=float(np.unique(1 - (1e-4 / nodelif.tau_mem))[0]),
            threshold=float(np.unique(nodelif.v_threshold)[0]),
            reset_mechanism="zero",
            all_to_all=True,
            linear_features=nodewrec.weight.shape[0],
            init_hidden=False,
        )
        self.lif1.recurrent.weight.data = torch.Tensor(nodewrec.weight)
        self.lif1.recurrent.bias.data = torch.Tensor(nodewrec.bias)

        node = nir_graph.nodes["fc2"]
        self.fc2 = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
        self.fc2.weight.data = torch.Tensor(node.weight)
        self.fc2.bias.data = torch.Tensor(node.bias)

        node = nir_graph.nodes["lif2"]
        self.lif2 = snn.Synaptic(
            alpha=float(np.unique(1 - (1e-4 / node.tau_syn))[0]),
            beta=float(np.unique(1 - (1e-4 / node.tau_mem))[0]),
            threshold=float(np.unique(node.v_threshold)[0]),
            reset_mechanism="zero",
            init_hidden=False,
        )

    def forward(self, x):
        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        spk1_rec = []  # not necessarily needed for inference
        spk2_rec = []

        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0)


# build initial network
###########################

print("\nload snnTorch module from checkpoint\n")

batch_size = 4
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
print("test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))

# export to NIR
###########################

print("\nexport to NIR graph\n")
nir_graph = export_nirtorch.to_nir(net, ds_test[0][0], ignore_dims=[0])
nir.write("braille_v2.nir", nir_graph)

# import from NIR - custom network
###########################

net0 = ImportedNetwork(nir_graph)

# import from NIR - using nirtorch
###########################

print("\nimport NIR graph (using nirtorch)\n")

nir_graph2 = nir.read("braille_v2.nir")

assert sorted(nir_graph2.nodes.keys()) == sorted(nir_graph.nodes.keys())
assert sorted(nir_graph2.edges) == sorted(nir_graph.edges)
for k in nir_graph.nodes:
    assert (
        nir_graph2.nodes[k].__class__.__name__ == nir_graph.nodes[k].__class__.__name__
    )
    for k2 in nir_graph.nodes[k].__dict__.keys():
        a = nir_graph.nodes[k].__dict__[k2]
        b = nir_graph2.nodes[k].__dict__[k2]
        if isinstance(a, np.ndarray):
            if not np.allclose(a, b):
                print("not close:", k, k2)
        elif isinstance(a, dict):
            for k3 in a:
                if not np.allclose(a[k3], b[k3]):
                    print("not close:", k, k2, k3)
        else:
            print("unknown type:", type(a), k, k2)

net2 = import_nirtorch.from_nir(nir_graph)

if check_parameters(net, net2):
    print("parameters match!")
else:
    print("parameters do not match!")

# forward pass through all networks in parallel
###########################

loader = DataLoader(ds_test, batch_size=4, shuffle=SHUFFLE, drop_last=False)
data, labels = next(iter(loader))

# reset network 0 states
spk1_0, syn1_0, mem1_0 = net0.lif1.init_rsynaptic()
syn2_0, mem2_0 = net0.lif2.init_synaptic()
# reset network 1 states
spk1, syn1, mem1 = net._modules["lif1"].init_rsynaptic()
syn2, mem2 = net._modules["lif2"].init_synaptic()
# reset network 2 states -- init_hidden=False
h2_state = nirtorch.from_nir.GraphExecutorState(
    state={
        "lif1": net2._modules["lif1"].init_rsynaptic(),  # 3-tuple: spk, syn, mem
        "lif2": net2._modules["lif2"].init_synaptic(),  # 2-tuple: syn, mem
    }
)

sout1_arr = []
sout2_arr = []
sout0_arr = []
for tstep in range(data.shape[1]):
    x = data[:, tstep, :]

    # forward pass through network 0 (custom, should work)
    cur1_0 = net0.fc1(x)
    spk1_0, syn1_0, mem1_0 = net0.lif1(cur1_0, spk1_0, syn1_0, mem1_0)
    cur2_0 = net0.fc2(spk1_0)
    spk2_0, syn2_0, mem2_0 = net0.lif2(cur2_0, syn2_0, mem2_0)
    sout0_arr.append(spk2_0)
    # forward pass through network 1
    cur1 = net._modules["fc1"](x)
    spk1, syn1, mem1 = net._modules["lif1"](cur1, spk1, syn1, mem1)
    cur2 = net._modules["fc2"](spk1)
    spk2, syn2, mem2 = net._modules["lif2"](cur2, syn2, mem2)
    sout1_arr.append(spk2)
    # forward pass through network 2
    spk_out, h2_state = net2(x, h2_state)
    sout2_arr.append(spk_out)

    comparison = np.array(
        [
            [
                cur1.sum().item(),
                syn1.sum().item(),
                mem1.sum().item(),
                spk1.sum().item(),
                cur2.sum().item(),
                syn2.sum().item(),
                mem2.sum().item(),
                spk2.sum().item(),
            ],
            [
                cur1_0.sum().item(),
                syn1_0.sum().item(),
                mem1_0.sum().item(),
                spk1_0.sum().item(),
                cur2_0.sum().item(),
                syn2_0.sum().item(),
                mem2_0.sum().item(),
                spk2_0.sum().item(),
            ],
            [
                h2_state.cache["fc1"].sum().item(),
                h2_state.state["lif1"][1].sum().item(),
                h2_state.state["lif1"][2].sum().item(),
                h2_state.cache["lif1"].sum().item(),
                h2_state.cache["fc2"].sum().item(),
                h2_state.state["lif2"][0].sum().item(),
                h2_state.state["lif2"][1].sum().item(),
                h2_state.cache["lif2"].sum().item(),
            ],
        ]
    )

    if not torch.equal(h2_state.cache["fc1"], cur1):
        print(tstep, "fc1", h2_state.cache["fc1"].sum().item(), cur1.sum().item())
    if not torch.equal(h2_state.cache["lif1"], spk1):
        print(tstep, "lif1", h2_state.cache["lif1"].sum().item(), spk1.sum().item())
    if not torch.equal(h2_state.cache["fc2"], cur2):
        print(tstep, "fc2", h2_state.cache["fc2"].sum().item(), cur2.sum().item())
    if not torch.equal(spk_out, spk2):
        print(tstep, "lif2", spk_out.sum().item(), spk2.sum().item())

    # print(tstep)
    # print(comparison)

print("\ntest the re-imported snnTorch network (using nirtorch)\n")
test_results = val_test_loop_nirtorch(
    ds_test, batch_size, net2, loss_fn, device, shuffle=SHUFFLE
)
print("test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))

# back to NIR and test
###########################

net2 = import_nirtorch.from_nir(nir_graph)  # reset the network
print("\nexporting back to NIR\n")
# HACK: initialize hidden state and pass it to the graph executor
model_fwd_args = [
    nirtorch.from_nir.GraphExecutorState(
        state={
            "lif1": net2._modules["lif1"].init_rsynaptic(),  # 3-tuple: spk, syn, mem
            "lif2": net2._modules["lif2"].init_synaptic(),  # 2-tuple: syn, mem
        }
    )
]
nir_graph2 = export_nirtorch.to_nir(
    net2, ds_test[0][0], model_fwd_args=model_fwd_args, ignore_dims=[0]
)
nir_graph2.infer_types()
nir.write("braille_v2a.nir", nir_graph2)
nir_graph = export_nirtorch.to_nir(
    net, ds_test[0][0], ignore_dims=[0]
)  # must reload (modified)

assert nir_graph.nodes.keys() == nir_graph2.nodes.keys(), "node keys mismatch"
for nodekey in nir_graph.nodes:
    a = (
        nir_graph.nodes[nodekey].__class__.__name__
        if nodekey in nir_graph.nodes
        else None
    )
    b = (
        nir_graph2.nodes[nodekey].__class__.__name__
        if nodekey in nir_graph2.nodes
        else None
    )
    assert a == b, f"node type mismatch: {a} vs {b}"
    for attr in nir_graph.nodes[nodekey].__dict__:
        close = None
        if isinstance(nir_graph.nodes[nodekey].__dict__[attr], np.ndarray):
            close = np.allclose(
                nir_graph.nodes[nodekey].__dict__[attr],
                nir_graph2.nodes[nodekey].__dict__[attr],
            )
        assert close is not False, f"node attribute mismatch: {nodekey}.{attr}"
