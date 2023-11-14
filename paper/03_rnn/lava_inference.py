
import nir
import nirtorch
from lava_rnn import from_nir
import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from snntorch import functional as SF
from snntorch import import_nirtorch
import matplotlib.pyplot as plt
import lava.lib.dl.slayer as slayer


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

nir_filename = 'braille_noDelay_bias_zero.nir'

nirgraph = nir.read(nir_filename)
net = from_nir(nirgraph)

nirgraph = nir.read(nir_filename)
net_snn = import_nirtorch.from_nir(nirgraph)

test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
letter_written = ['Space', 'A', 'E', 'I', 'O', 'U', 'Y']
loss_fn = SF.ce_count_loss()

batch_size = 64
shuffle = False
loader = DataLoader(ds_test, batch_size=batch_size, shuffle=shuffle)


def val_test_loop_nirtorch(dataset, batch_size, net, loss_fn, shuffle=True):
    with torch.no_grad():
        net.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        batch_loss = []
        batch_acc = []

        for data, labels in loader:
            data = data.swapaxes(1, 0)
            labels = labels

            h_state = nirtorch.from_nir.GraphExecutorState(
                state={
                    'lif1': net._modules['lif1'].init_rsynaptic(),  # 3-tuple: spk, syn, mem
                    'lif2': net._modules['lif2'].init_synaptic(),  # 2-tuple: syn, mem
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
            batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())

        return [np.mean(batch_loss), np.mean(batch_acc)]


acc = val_test_loop_nirtorch(ds_test, batch_size, net_snn, loss_fn, shuffle=False)[1]
print(f'snnTorch test accuracy: {acc:.2%}')

wg = nirgraph.nodes['fc1'].weight
wn = net._modules['fc1'].weight.detach().squeeze(-1).squeeze(-1).squeeze(-1).numpy()
bg = nirgraph.nodes['fc1'].bias
bn = net._modules['fc1'].bias.detach().numpy()
print('weights close', np.allclose(wg, wn), np.allclose(bg, bn))

[e.elem for e in net.get_execution_order()]

# reset the states before the first batch is passed
nirgraph = nir.read(nir_filename)
net = from_nir(nirgraph)

spk1_lava_over_snn = []

with torch.no_grad():
    net.eval()
    net_snn.eval()

    batch_loss = []
    batch_acc = []
    batch_acc_snn = []
    pred = []
    act_out = []
    for batch_idx, (data, labels) in enumerate(loader):  # data comes as: NTC
        # print('new batch')
        data_ldl = data.swapaxes(1, 2)  # NCT
        data_snn = data.swapaxes(1, 0)  # TNC

        #####
        # lava-dl network
        x = data_ldl
        int_lava = {}
        rec_hid = {}

        for node in net.get_execution_order():
            if isinstance(node.elem, (slayer.block.cuba.Recurrent, slayer.block.cuba.Dense)):
                if not torch.equal(node.elem.neuron.current_state, torch.Tensor([0])):
                    print('current_state not zero, resetting manually')
                    node.elem.neuron.current_state = torch.Tensor([0])
                if not torch.equal(node.elem.neuron.voltage_state, torch.Tensor([0])):
                    print('voltage_state not zero, resetting manually')
                    node.elem.neuron.voltage_state = torch.Tensor([0])
                assert torch.equal(node.elem.neuron.current_state, torch.Tensor([0]))
                assert torch.equal(node.elem.neuron.voltage_state, torch.Tensor([0]))
            x = node.elem(x)
            if isinstance(x, tuple):
                x, v, c = x
                rec_hid[node.name] = {'v': v, 'c': c}
            int_lava[node.name] = x
        # spk_out, hid_rec = net(data_ldl)
        # spk_out = spk_out.moveaxis(2, 0)  # TCN

        #####
        # snnTorch network
        h_state = nirtorch.from_nir.GraphExecutorState(
            state={
                'lif1': net_snn._modules['lif1'].init_rsynaptic(),  # 3-tuple: spk, syn, mem
                'lif2': net_snn._modules['lif2'].init_synaptic(),  # 2-tuple: syn, mem
            }
        )
        spk_out_arr = []
        h_state_arr = []
        for t in range(data_snn.shape[0]):
            spk_out_snn, h_state = net_snn(data_snn[t], h_state)
            spk_out_arr.append(spk_out_snn)
            h_state_arr.append(h_state)
        spk_out_arr = torch.stack(spk_out_arr, dim=0)
        snntorch_cache = {
            k: torch.stack([h_state.cache[k] for h_state in h_state_arr], dim=-1)
            for k in h_state.cache.keys()
        }
        snntorch_curr = {
            k: torch.stack([h_state.state[k][1 if len(h_state.state[k]) == 3 else 0]
                            for h_state in h_state_arr], dim=-1)
            for k in h_state.state.keys()
        }
        snntorch_mem = {
            k: torch.stack([h_state.state[k][2 if len(h_state.state[k]) == 3 else 1]
                            for h_state in h_state_arr], dim=-1)
            for k in h_state.state.keys()
        }

        # net_snn.graph.node_map_by_id['lif1'].elem.recurrent.weight
        # net.graph.node_map_by_id['lif1'].elem.recurrent_synapse.weight.squeeze(-1).squeeze(-1).squeeze(-1)
        # torch.allclose(net.graph.node_map_by_id['lif1'].elem.recurrent_synapse.weight.squeeze(-1).squeeze(-1).squeeze(-1), net_snn.graph.node_map_by_id['lif1'].elem.recurrent.weight)
        # torch.allclose(torch.eye(38), net.graph.node_map_by_id['lif1'].elem.input_synapse.weight.squeeze(-1).squeeze(-1).squeeze(-1))

        #####
        # analyze
        fc1_lava = int_lava['fc1']
        fc1_snntorch = snntorch_cache['fc1']
        mem1_lava = rec_hid['lif1']['v']
        mem1_snntorch = snntorch_mem['lif1']
        cur1_lava = rec_hid['lif1']['c']
        cur1_snntorch = snntorch_curr['lif1']
        spk1_lava = int_lava['lif1']
        spk1_snntorch = snntorch_cache['lif1']

        fig, axs = plt.subplots(4, 2, figsize=(24, 8), dpi=200, sharex=True)
        axs[0][0].set_title('fc1 output traces - lava-dl')
        axs[0][0].set_xlim(0, data.shape[1])
        axs[0][0].plot(fc1_lava[0].T)
        axs[1][0].set_title('fc1 output traces - snnTorch')
        axs[1][0].set_xlim(0, data.shape[1])
        axs[1][0].plot(fc1_snntorch[0].T)

        axs[2][0].set_title('lif1 current - lava-dl')
        axs[2][0].set_xlim(0, data.shape[1])
        axs[2][0].plot(cur1_lava[0].T)
        axs[3][0].set_title('lif1 current - snnTorch')
        axs[3][0].set_xlim(0, data.shape[1])
        axs[3][0].plot(cur1_snntorch[0].T)

        axs[0][1].set_title('lif1 membrane - lava-dl')
        # axs[2].set_ylim(-1, 1.2)
        axs[0][1].set_xlim(0, data.shape[1])
        axs[0][1].plot(mem1_lava[0].T)
        axs[1][1].set_title('lif1 membrane - snnTorch')
        # axs[3].set_ylim(-1, 1.2)
        axs[1][1].set_xlim(0, data.shape[1])
        axs[1][1].plot(mem1_snntorch[0].T)

        axs[2][1].set_title('lif1 spikes - lava-dl')
        axs[2][1].set_xlim(0, data.shape[1])
        for idx, yt_idx in enumerate(spk1_lava[0]):
            axs[2][1].eventplot(np.where(yt_idx == 1)[0], lineoffsets=idx, linelengths=0.8)
        axs[3][1].set_title('lif1 spikes - snnTorch')
        axs[3][1].set_xlim(0, data.shape[1])
        for idx, yt_idx in enumerate(spk1_snntorch[0]):
            axs[3][1].eventplot(np.where(yt_idx == 1)[0], lineoffsets=idx, linelengths=0.8)
        plt.tight_layout()
        plt.savefig('lava_analysis_nir.png')
        plt.close()

        spk1_lava_over_snn.append((spk1_lava.sum() / spk1_snntorch.sum()).item())
        print(spk1_lava_over_snn)

        # lava-dl loss & accuracy
        spk_out = int_lava['lif2'].moveaxis(2, 0)  # TBN
        loss_val = loss_fn(spk_out, labels)
        batch_loss.append(loss_val.detach().cpu().item())
        act_total_out = torch.sum(spk_out, 0)  # sum over time
        _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax output > labels
        pred.extend(neuron_max_act_total_out.detach().cpu().numpy())
        act_out.extend(act_total_out.detach().cpu().numpy())
        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())

        # snntorch accuracy
        spk_out_snn = spk_out_arr
        act_total_out_snn = torch.sum(spk_out_snn, 0)  # sum over time
        _, neuron_max_act_total_out_snn = torch.max(act_total_out_snn, 1)
        batch_acc_snn.extend((neuron_max_act_total_out_snn == labels).detach().cpu().numpy())

        if batch_idx == 0:
            print('saving activity for first sample')
            fname = 'lava_activity_noDelay_bias_zero.npy'
            np.save(fname, spk1_lava[0].detach().numpy())

    test_results = [np.mean(batch_loss), np.mean(batch_acc)]

print(f"lava-dl test accuracy: {test_results[1]:.2%}")
print(f"snntorch test accuracy: {np.mean(batch_acc_snn):.2%}")

fname = 'lava_accuracy_noDelay_bias_zero.npy'
np.save(fname, np.mean(batch_acc))
