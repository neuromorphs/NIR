import nir
from _lava_rnn import from_nir as from_nir_to_lava
import torch
import numpy as np
from torch.utils.data import DataLoader
from snntorch import functional as SF
import lava.lib.dl.slayer as slayer


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

nir_filename = "braille_noDelay_bias_zero.nir"
nirgraph = nir.read(nir_filename)
net = from_nir_to_lava(nirgraph)

test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
letter_written = ["Space", "A", "E", "I", "O", "U", "Y"]
loss_fn = SF.ce_count_loss()

batch_size = 64
shuffle = False
loader = DataLoader(ds_test, batch_size=batch_size, shuffle=shuffle)

# reset the states before the first batch is passed
nirgraph = nir.read(nir_filename)
net = from_nir_to_lava(nirgraph)

with torch.no_grad():
    net.eval()
    batch_loss = []
    batch_acc = []
    for batch_idx, (data, labels) in enumerate(loader):  # data comes as: NTC
        data_ldl = data.swapaxes(1, 2)  # NCT

        x = data_ldl
        int_lava = {}
        rec_hid = {}

        for node in net.get_execution_order():
            if isinstance(
                node.elem, (slayer.block.cuba.Recurrent, slayer.block.cuba.Dense)
            ):
                if not torch.equal(node.elem.neuron.current_state, torch.Tensor([0])):
                    print("current_state not zero, resetting manually")
                    node.elem.neuron.current_state = torch.Tensor([0])
                if not torch.equal(node.elem.neuron.voltage_state, torch.Tensor([0])):
                    print("voltage_state not zero, resetting manually")
                    node.elem.neuron.voltage_state = torch.Tensor([0])
                assert torch.equal(node.elem.neuron.current_state, torch.Tensor([0]))
                assert torch.equal(node.elem.neuron.voltage_state, torch.Tensor([0]))
            x = node.elem(x)
            int_lava[node.name] = x

        fc1_lava = int_lava["fc1"]
        spk1_lava = int_lava["lif1"]

        # lava-dl loss & accuracy
        spk_out = int_lava["lif2"].moveaxis(2, 0)  # TBN
        loss_val = loss_fn(spk_out, labels)
        batch_loss.append(loss_val.detach().cpu().item())
        act_total_out = torch.sum(spk_out, 0)  # sum over time
        _, neuron_max_act_total_out = torch.max(
            act_total_out, 1
        )  # argmax output > labels

        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())

        if batch_idx == 0:
            print("saving activity for first sample")
            fname = "lava_activity_noDelay_bias_zero.npy"
            np.save(fname, spk1_lava[0].detach().numpy())

    test_results = [np.mean(batch_loss), np.mean(batch_acc)]

print(f"lava-dl test accuracy: {test_results[1]:.2%}")
fname = "lava_accuracy_noDelay_bias_zero.npy"
np.save(fname, np.mean(batch_acc))
