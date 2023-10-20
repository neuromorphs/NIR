# from collections import defaultdict

# import matplotlib.pyplot as plt

# import norse
import numpy as np
import tonic
import torch
import tqdm
from spinnaker2 import hardware, s2_nir, snn

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

s2_nir.model_summary(nir_graph)

print("\nInferring types.")
nir_graph.infer_types()


print("\nDone")
s2_nir.model_summary(nir_graph)
# nir_graph = s2_nir.replace_sumpool2d_by_sumpool2d_if(nir_graph)
# s2_nir.model_summary(nir_graph)
# s2_nir.add_output_to_node('1', nir_graph, '1_out')
# s2_nir.add_output_to_node('3', nir_graph, '3_out')
# s2_nir.add_output_to_node('6', nir_graph, '6_out')
# s2_nir.add_output_to_node('10', nir_graph, '10_out')

print("\nAdding outputs for all IFs")
s2_nir.model_summary(nir_graph)

net, inp, outp = s2_nir.from_nir(
    nir_graph,
    outputs=["v", "spikes"],
    discretization_timestep=0.0001,
    conn_delay=0,
    scale_weights=True,
)


pop_order = [5, 0, 3, 4, 1, 2]
proj_order = [1, 0, 2, 3, 4]
for pop_id in pop_order:
    pop = net.populations[pop_id]
    # if pop.name == "1":
    #    pop.set_max_atoms_per_core(128)
    if pop.name == "3":
        pop.set_max_atoms_per_core(256)
    if pop.name == "6":
        pop.set_max_atoms_per_core(128)
    if pop.name == "10":
        pop.set_max_atoms_per_core(16)
    print(f"{pop.name}: {pop.size}")
for proj_id in proj_order:
    proj = net.projections[proj_id]
    if isinstance(proj, snn.Projection):
        print(
            f"{proj.pre.name} -> {proj.post.name}: {proj.sparse_weights.nonzero()[0].shape[0]}"
        )
    elif isinstance(proj, snn.Conv2DProjection):
        print(f"{proj.pre.name} -> {proj.post.name}: {proj.weights.shape[0]}")


to_frame = tonic.transforms.ToFrame(
    sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1e3
)
dataset = tonic.datasets.NMNIST(".", transform=to_frame, train=False)
loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
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
    net.reset()
    hw.run(net, timesteps, sys_tick_in_s=2.5e-3, debug=False)
    # spike_times = outp[0].get_spikes()
    # print([(i,o.name) for i,o in enumerate(outp)])
    voltages = outp[0].get_voltages()
    out_v = np.zeros(10)
    # plt.figure()
    for idx, v in voltages.items():
        # plt.plot(v, label=f"{idx}")
        out_v[idx] = v[-1]

    out_spikes = np.zeros(10)
    out_spike_times = outp[0].get_spikes()
    for idx, spikes in out_spike_times.items():
        out_spikes[idx] = len(spikes) / x.shape[0]

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
                spike_idx.append(np.ones_like(spike_times[i])*i)
                spike_t.append(np.array(spike_times[i]))
        if len(spike_idx) > 0:
            plt.scatter(np.concatenate(spike_t), np.concatenate(spike_idx), marker='.', s=1)
    
"""
    # plt.show()
    """    #plt.pause(0.1)
"""
    return np.argmax(out_v), np.argmax(out_spikes)
    # return -1, np.argmax(out_spikes)


# hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.48")  # use ethernet
# hw = brian2_sim.Brian2Backend()

predicted_v = []
predicted_spike = []
groundtruth = []
with torch.no_grad():
    for batch in tqdm.tqdm(loader):
        x, y = batch
        x = x.detach().numpy()
        y = y.detach().numpy()
        print(" ")
        hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.48")  # use ethernet
        pred_v, pred_spike = evaluate(hw, net, inp, outp, x, y)
        del hw
        predicted_v.append(pred_v)
        predicted_spike.append(pred_spike)
        groundtruth.append(y[0])
        accuracy_v = np.mean(np.array(predicted_v) == np.array(groundtruth))
        accuracy_spike = np.mean(np.array(predicted_spike) == np.array(groundtruth))
        print(f"Got: v:{pred_v}, spike:{pred_spike}, ground truth: {y[0]}")
        print(
            f"Current accuracy by voltages/spike: {accuracy_v:.4f}/{accuracy_spike:.4f}"
            f" after {len(predicted_v)}"
        )
