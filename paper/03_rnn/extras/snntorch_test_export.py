import json
import numpy as np
import snntorch as snn
import argparse
from snntorch import functional as SF
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nir

# NOTE: this requires snntorch/nir (PR) and nirtorch/master (unreleased)
from snntorch import export_nirtorch


device = torch.device("cpu")

# parser = argparse.ArgumentParser()
# parser.add_argument("model", type=str, help="model name")
# args = parser.parse_args()
# saved_state_dict_path = f"data/model_ref_{args.model}.pt"
# best_val_layers = torch.load(saved_state_dict_path, map_location=device)
# parameters_path = f"data/parameters_ref_{args.model}.json"
# model_name = args.model

# model_name = "retrained_zero"
model_name = "retrained_nobias_subtract"
# saved_state_dict_path = "data/retrained_snntorch_20231024_110806.pt"
saved_state_dict_path = "data/model_noDelay_noBias_ref_subtract.pt"
best_val_layers = torch.load(saved_state_dict_path, map_location=device)
# parameters_path = "data/parameters_ref_zero.json"
parameters_path = "data/parameters_noDelay_noBias_ref_subtract.json"

bias = False

with open(parameters_path) as f:
    parameters = json.load(f)
regularization = [parameters["reg_l1"], parameters["reg_l2"]]

loss_fn = SF.ce_count_loss()

test_data_path = "data/ds_test.pt"
ds_test = torch.load(test_data_path)
SHUFFLE = False


def model_build(settings, input_size, num_steps, device, bias=True):
    input_channels = int(input_size)
    num_hidden = int(settings["nb_hidden"])
    num_outputs = 7
    spike_grad = surrogate.fast_sigmoid(slope=int(settings["slope"]))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(input_channels, num_hidden, bias=bias)
            self.lif1 = snn.RSynaptic(
                alpha=settings["alpha_r"],
                beta=settings["beta_r"],
                linear_features=num_hidden,
                spike_grad=spike_grad,
                reset_mechanism="zero",
                reset_delay=False,
            )
            if not bias:
                self.lif1.recurrent.bias = None
            self.fc2 = nn.Linear(num_hidden, num_outputs, bias=bias)
            self.lif2 = snn.Synaptic(
                alpha=settings["alpha_out"],
                beta=settings["beta_out"],
                spike_grad=spike_grad,
                reset_mechanism="zero",
                reset_delay=False,
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


# build initial network
###########################

print("\nload snnTorch module from checkpoint\n")
batch_size = 4
input_size = 12
num_steps = next(iter(ds_test))[0].shape[0]
net = model_build(parameters, input_size, num_steps, device, bias)
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
nir.write(f"braille_{model_name}.nir", nir_graph)
