import nir
import torch
import numpy as np
# import os
# import torch.nn as nn
from torch.utils.data import DataLoader
from snntorch import functional as SF

# nirtorch loading for lava-dl and snntorch
from lava_rnn import from_nir
from snntorch import import_nirtorch
from nirtorch.from_nir import GraphExecutorState

run_only_one_batch = False

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

nirgraph = nir.read('braille.nir')
net = from_nir(nirgraph)

nirgraph = nir.read('braille.nir')
net_snn = import_nirtorch.from_nir(nirgraph)

test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
loss_fn = SF.ce_count_loss()

batch_size = 64

# snntorch
#########################

snn_inputs = []
loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    net_snn.eval()
    batch_loss = []
    batch_acc = []

    for data, labels in loader:
        data = data.swapaxes(1, 0)
        snn_inputs.append(data.swapaxes(1, 0))
        labels = labels

        h_state = GraphExecutorState(state={
                'lif1': net_snn._modules['lif1'].init_rsynaptic(),  # 3-tuple: spk, syn, mem
                'lif2': net_snn._modules['lif2'].init_synaptic(),  # 2-tuple: syn, mem
            }
        )

        spk_out_arr = []
        h_states = []
        for t in range(data.shape[0]):
            spk_out, h_state = net_snn(data[t], h_state)
            spk_out_arr.append(spk_out)
            h_states.append(h_state)
        spk_out = torch.stack(spk_out_arr, dim=0)

        loss_val = loss_fn(spk_out, labels)
        batch_loss.append(loss_val.detach().cpu().item())
        act_total_out = torch.sum(spk_out, 0)
        _, neuron_max_act_total_out = torch.max(act_total_out, 1)
        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())
        if run_only_one_batch:
            break
    test_results = [np.mean(batch_loss), np.mean(batch_acc)]
print("Test accuracy (snnTorch): {}%".format(np.round(test_results[1] * 100, 2)))

# lava-dl
#########################

ldl_inputs = []
loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    net.eval()
    batch_loss = []
    batch_acc = []
    for data, labels in loader:
        data = data.swapaxes(1, 2)  # NCT
        ldl_inputs.append(data.swapaxes(1, 2))

        spk_out, hid_rec = net(data)
        spk_out = spk_out.moveaxis(2, 0)  # TCN

        loss_val = loss_fn(spk_out, labels)
        batch_loss.append(loss_val.detach().cpu().item())
        act_total_out = torch.sum(spk_out, 0)
        _, neuron_max_act_total_out = torch.max(act_total_out, 1)
        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())
        if run_only_one_batch:
            break
    test_results = [np.mean(batch_loss), np.mean(batch_acc)]
print("Test accuracy (lava-dl): {}%".format(np.round(test_results[1] * 100, 2)))

# compare inputs
#########################

for i in range(len(snn_inputs)):
    print(f'input #{i} match: {torch.allclose(snn_inputs[i], ldl_inputs[i])}')

fc1_snn = torch.stack([h.cache['fc1'] for h in h_states], dim=-1)
fc1_ldl = hid_rec.cache['fc1']
print(f'fc1 match (atol=1e-6): {torch.allclose(fc1_snn, fc1_ldl, atol=1e-6)}')
print(f'fc1 match (atol=1e-8): {torch.allclose(fc1_snn, fc1_ldl, atol=1e-8)}')

lif1_snn = torch.stack([h.cache['lif1'] for h in h_states], dim=-1)
lif1_ldl = hid_rec.cache['lif1']

# lif2_ldl = hid_rec.cache['lif2']
# net._modules['lif1'].neuron.current_decay
# net._modules['lif1'].neuron.voltage_decay

# check if lif1 synapse weights are identity matrix
torch.allclose(net._modules['lif1'].input_synapse.weight.flatten(1).detach(), torch.eye(38))

# fc2_snn = torch.stack([h.cache['fc2'] for h in h_states], dim=-1)
# fc2_ldl = hid_rec.cache['fc2']
# print(f'fc2 match (atol=1e-2): {torch.allclose(fc2_snn, fc2_ldl, atol=1e-2)}')
# print(f'fc2 match (atol=1e-6): {torch.allclose(fc2_snn, fc2_ldl, atol=1e-6)}')
# print(f'fc2 match (atol=1e-8): {torch.allclose(fc2_snn, fc2_ldl, atol=1e-8)}')
print('done')
