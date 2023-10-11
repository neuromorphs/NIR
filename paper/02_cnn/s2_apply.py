# from collections import defaultdict

# import matplotlib.pyplot as plt
# import norse
import numpy as np
import tonic
import torch
import tqdm
from spinnaker2 import brian2_sim, s2_nir  # , hardware, helpers

import nir

nir_graph = nir.read("scnn_mnist.nir")
nlist = (
    [
        "input",
    ]
    + [str(i) for i in range(13)]
    + [
        "output",
    ]
)
n = nir_graph.nodes
# for k in nlist:
#    with np.printoptions(threshold=20, edgeitems=1):
#        #print((k, type(n[k]), n[k]))
#        print((k, type(n[k])), n[k].input_type, n[k].output_type)

n["0"].input_type["input"] = n["input"].output_type["output"]
n["0"].output_type["output"] = n["1"].input_type["input"]

n["2"].input_type["input"] = n["1"].output_type["output"]
n["2"].output_type["output"] = n["3"].input_type["input"]

n["4"].input_type["input"] = n["3"].output_type["output"]
tmp = np.copy(n["4"].input_type["input"])
tmp[2:] = tmp[2:] / 2
n["4"].output_type["output"] = tmp

n["5"].input_type["input"] = n["4"].output_type["output"]
n["5"].output_type["output"] = n["6"].input_type["input"]

n["7"].input_type["input"] = n["6"].output_type["output"]
tmp = np.copy(n["7"].input_type["input"])
tmp[2:] = tmp[2:] / 2
n["7"].output_type["output"] = tmp

n["8"].input_type["input"] = n["7"].output_type["output"]
n["8"].output_type["output"] = n["9"].input_type["input"]
"""Def replace_sumpool2d_by_lif(nir_graph):

for n in nir_graph.nodes:     if isinstance(n, SumPool2d):         name = n.name
"""

# remove all batch dimensions
for k in n.keys():
    input_shape = n[k].input_type["input"]
    n[k].input_type["input"] = input_shape[input_shape != 1]
    output_shape = n[k].output_type["output"]
    n[k].output_type["output"] = output_shape[output_shape != 1]

s2_nir.model_summary(nir_graph)
# nir_graph = s2_nir.replace_sumpool2d_by_sumpool2d_if(nir_graph)
# s2_nir.model_summary(nir_graph)

net, inp, outp = s2_nir.from_nir(
    nir_graph, outputs=["v", "spikes"], discretization_timestep=0.0001, conn_delay=0
)

pop_order = [5, 0, 3, 4, 1, 2]
proj_order = [1, 0, 2, 3, 4]
for pop_id in pop_order:
    pop = net.populations[pop_id]
    if pop.name == "1":
        pop.set_max_atoms_per_core(128)
    if pop.name == "3":
        pop.set_max_atoms_per_core(32)
    if pop.name == "6":
        pop.set_max_atoms_per_core(32)
    if pop.name == "10":
        pop.set_max_atoms_per_core(24)
    print(f"{pop.name}: {pop.size}")
for proj_id in proj_order:
    proj = net.projections[proj_id]
    print(
        f"{proj.pre.name} -> {proj.post.name}: {proj.sparse_weights.nonzero()[0].shape[0]}"
    )

to_frame = tonic.transforms.ToFrame(
    sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1e3
)
dataset = tonic.datasets.NMNIST(".", transform=to_frame, train=False)
loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
)


def convert_input(x):
    print(x.shape)
    d = {}
    # T = x.shape[0]
    C = x.shape[2]
    H = x.shape[3]
    W = x.shape[4]
    for c in range(C):
        for h in range(H):
            for w in range(W):
                d[c * H * W + h * W + w] = x[:, 0, c, h, w].nonzero()[0].tolist()
    return d


def evaluate(hw, net, inp, outp, x, y):
    input_spikes = convert_input(x)
    inp[0].params = input_spikes

    timesteps = x.shape[0] + 1
    hw.run(net, timesteps)
    # spike_times = outp[0].get_spikes()
    voltages = outp[0].get_voltages()
    out_v = np.zeros(10)
    # plt.figure()
    for idx, v in voltages.items():
        # plt.plot(v, label=f"{idx}")
        out_v[idx] = v[-1]
    # plt.title(f"output: {np.argmax(out_v)}, correct: {y}")
    # plt.legend()
    """
    for pop in net.populations:
        spike_times = pop.get_spikes()
        plt.figure()
        plt.title(pop.name)
        spike_idx = []
        spike_t = []
        for i in spike_times:
            if len(spike_times[i])>0:
                spike_idx.append(np.ones_like(spike_times[i]))
                spike_t.append(np.array(spike_times[i]))
        if len(spike_idx) > 0:
            plt.scatter(np.concatenate(spike_t), np.concatenate(spike_idx), marker='.')
    """

    # plt.show()
    # plt.pause(0.1)

    print("done running")
    return np.argmax(out_v)


# hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.48")  # use ethernet
hw = brian2_sim.Brian2Backend()

losses = []
accuracies = []
predicted = []
groundtruth = []
with torch.no_grad():
    for batch in tqdm.tqdm(loader):
        x, y = batch
        x = x.detach().numpy()
        y = y.detach().numpy()
        pred = evaluate(hw, net, inp, outp, x, y)
        predicted.append(pred)
        groundtruth.append(y)
        accuracy = np.mean(np.array(predicted) == np.array(groundtruth))
        print(f"Current accuracy: {accuracy:.4f} after {len(predicted)}")
