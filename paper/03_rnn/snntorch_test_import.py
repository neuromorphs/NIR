import numpy as np
from snntorch import functional as SF
import torch
from torch.utils.data import DataLoader
import nir
# NOTE: this requires snntorch/nir (PR) and nirtorch/master (unreleased)
from snntorch import import_nirtorch
import nirtorch


device = torch.device("cpu")
loss_fn = SF.ce_count_loss()
test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
SHUFFLE = False


def val_test_loop_nirtorch(dataset, batch_size, net, loss_fn, device, shuffle=True):
    with torch.no_grad():
        net.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        batch_loss = []
        batch_acc = []

        for data, labels in loader:
            data = data.to(device).swapaxes(1, 0)
            labels = labels.to(device)

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


# import from NIR - using nirtorch
###########################

nir_graph = nir.read('braille_v2.nir')
net2 = import_nirtorch.from_nir(nir_graph)
test_results = val_test_loop_nirtorch(ds_test, 64, net2, loss_fn, device, shuffle=SHUFFLE)
print("test accuracy: {}%".format(np.round(test_results[1] * 100, 2)))
