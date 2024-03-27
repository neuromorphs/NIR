import copy
import datetime
import json
import os
import numpy as np
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# lava-dl
import nir
import nirtorch
from lava_rnn import from_nir
import lava.lib.dl.slayer as slayer
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

store_weights = True
training_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cpu")
# device = torch.device("mps")

# DEVICE SETTINGS
use_gpu = False

# LOAD DATA
ds_train = torch.load("data/ds_train.pt")
ds_val = torch.load("data/ds_train.pt")
ds_test = torch.load("data/ds_test.pt")

letter_written = ["Space", "A", "E", "I", "O", "U", "Y"]

# load the lava-dl network
nir_filename = 'braille_noDelay_bias_zero.nir'
nirgraph = nir.read(nir_filename)
net = from_nir(nirgraph).to(device)

batch_size = 64
train_loader = DataLoader(
    ds_train, batch_size=batch_size, shuffle=True, drop_last=False
)
error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

n_epochs = 300
stats_acc = []
stats_lss = []

for epoch in range(n_epochs):
    n_samples = 0
    n_correct = 0
    sum_loss = 0.
    for data, labels in train_loader:
        data = data.to(device).swapaxes(1, 2)
        labels = labels.to(device)

        net.train()
        spk_rec, hid_rec = net(data)

        # regularization = 0.0006 * torch.mean(torch.sum(hid_rec.cache['lif1'], -1))
        # regularization += 0.000004 * (hid_rec.cache['lif1'].sum(2).sum(0) ** 2).mean()
        regularization = 0
        loss = error(spk_rec, labels) + regularization * 0.01
        # spk_rec = spk_rec.moveaxis(2, 0)  # TCN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_samples += data.shape[0]
        sum_loss += loss.detach().cpu().data.item() * spk_rec.shape[0]
        n_correct += torch.sum(
            slayer.classifier.Rate.predict(spk_rec) == labels
        ).detach().cpu().data.item()
    batch_acc = n_correct/n_samples
    batch_loss = sum_loss/n_samples
    stats_acc.append(batch_acc)
    stats_lss.append(batch_loss)
    print(f"Epoch [{epoch+1:3}/{n_epochs:3}] loss {batch_loss:.3f} accuracy {batch_acc:.2%}")

input('done...')

loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    net.eval()

    batch_loss = []
    batch_acc = []
    pred = []
    act_out = []
    for data, labels in loader:  # data comes as: NTC
        data_ldl = data.swapaxes(1, 2)  # NCT
        spk_out, hid_rec = net(data_ldl)
        spk_out = spk_out.moveaxis(2, 0)  # TCN
        #####
        act_total_out = torch.sum(spk_out, 0)  # sum over time
        _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax output > labels
        pred.extend(neuron_max_act_total_out.detach().cpu().numpy())
        act_out.extend(act_total_out.detach().cpu().numpy())
        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())
        # batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy()))

    test_results = [np.mean(batch_loss), np.mean(batch_acc)]

print(f"lava-dl test accuracy: {test_results[1]:.2%}")


def training_loop(
    dataset, batch_size, net, optimizer, loss_fn, device, regularization=None
):
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    batch_loss = []
    batch_acc = []

    for data, labels in train_loader:
        data = data.to(device).swapaxes(1, 0)
        labels = labels.to(device)

        net.train()
        # spk_rec, _, _, hid_rec = net(data)
        spk_rec, hid_rec = net(data)

        # Training loss
        # "reg_l1": 0.0006,
        # "reg_l2": 0.000004,
        if regularization is not None:
            # L1 loss on spikes per neuron from the hidden layer
            reg_loss = regularization[0] * torch.mean(torch.sum(hid_rec, 0))
            # L2 loss on total number of spikes from the hidden layer
            reg_loss = reg_loss + regularization[1] * torch.mean(
                torch.sum(torch.sum(hid_rec, dim=0), dim=1) ** 2
            )
            loss_val = loss_fn(spk_rec, labels) + reg_loss
        else:
            loss_val = loss_fn(spk_rec, labels)

        batch_loss.append(loss_val.detach().cpu().item())

        # Training accuracy
        act_total_out = torch.sum(spk_rec, 0)  # sum over time
        _, neuron_max_act_total_out = torch.max(
            act_total_out, 1
        )  # argmax over output units to compare to labels
        batch_acc.extend(
            (neuron_max_act_total_out == labels).detach().cpu().numpy()
        )
        # the "old" one with mean per batch:
        # batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy()))

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    epoch_loss = np.mean(batch_loss)
    epoch_acc = np.mean(batch_acc)

    return [epoch_loss, epoch_acc]


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
            # batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy()

        if label_probabilities:
            log_softmax_fn = nn.LogSoftmax(dim=-1)
            log_p_y = log_softmax_fn(act_total_out)
            return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y)
        else:
            return [np.mean(batch_loss), np.mean(batch_acc)]


# PREPARE FOR TRAINING

num_epochs = 500

batch_size = 64

input_size = 12
num_steps = next(iter(ds_test))[0].shape[0]

net = model_build(parameters, input_size, num_steps, device)

loss_fn = SF.ce_count_loss()

optimizer = torch.optim.Adam(net.parameters(), lr=parameters["lr"], betas=(0.9, 0.999))


# TRAINING (with validation and test)

print(
    "Training started on: {}-{}-{} {}:{}:{}\n".format(
        training_datetime[:4],
        training_datetime[4:6],
        training_datetime[6:8],
        training_datetime[-6:-4],
        training_datetime[-4:-2],
        training_datetime[-2:],
    )
)

training_results = []
validation_results = []

for ee in range(num_epochs):
    train_loss, train_acc = training_loop(
        ds_train,
        batch_size,
        net,
        optimizer,
        loss_fn,
        device,
        regularization=regularization,
    )
    val_loss, val_acc = val_test_loop(
        ds_val, batch_size, net, loss_fn, device, regularization=regularization
    )

    training_results.append([train_loss, train_acc])
    validation_results.append([val_loss, val_acc])

    if (ee == 0) | ((ee + 1) % 10 == 0):
        print(f"\tepoch {ee + 1}/{num_epochs} done \t --> \ttraining accuracy (loss): "
              f"{np.round(training_results[-1][1] * 100, 4)}% "
              f"({training_results[-1][0]}), \tvalidation accuracy (loss): "
              f"{np.round(validation_results[-1][1] * 100, 4)}% ({validation_results[-1][0]})")

    if val_acc >= np.max(np.array(validation_results)[:, 1]):
        best_val_layers = copy.deepcopy(net.state_dict())


training_hist = np.array(training_results)
validation_hist = np.array(validation_results)

# best training and validation at best training
acc_best_train = np.max(training_hist[:, 1])
epoch_best_train = np.argmax(training_hist[:, 1])
acc_val_at_best_train = validation_hist[epoch_best_train][1]

# best validation and training at best validation
acc_best_val = np.max(validation_hist[:, 1])
epoch_best_val = np.argmax(validation_hist[:, 1])
acc_train_at_best_val = training_hist[epoch_best_val][1]

print("\n")
print("Overall results:")
print(
    "\tBest training accuracy: {}% ({}% corresponding validation accuracy) at epoch {}/{}".format(
        np.round(acc_best_train * 100, 4),
        np.round(acc_val_at_best_train * 100, 4),
        epoch_best_train + 1,
        num_epochs,
    )
)
print(
    "\tBest validation accuracy: {}% ({}% corresponding training accuracy) at epoch {}/{}".format(
        np.round(acc_best_val * 100, 4),
        np.round(acc_train_at_best_val * 100, 4),
        epoch_best_val + 1,
        num_epochs,
    )
)
print("\n")

# Test
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
print("Test accuracy: {}%\n".format(np.round(test_results[1] * 100, 2)))

# Ns single-sample inferences to check label probabilities
Ns = 10
for ii in range(Ns):
    single_sample = next(iter(DataLoader(ds_test, batch_size=1, shuffle=True)))
    _, lbl_probs = val_test_loop(
        TensorDataset(single_sample[0], single_sample[1]),
        1,
        net,
        loss_fn,
        device,
        shuffle=False,
        saved_state_dict=best_val_layers,
        label_probabilities=True,
        regularization=regularization,
    )
    print("Single-sample inference {}/{} from test set:".format(ii + 1, Ns))
    print(
        "Sample: {} \tPrediction: {}".format(
            letter_written[single_sample[1]],
            letter_written[torch.max(lbl_probs.cpu(), 1)[1]],
        )
    )
    print(
        "Label probabilities (%): {}\n".format(
            np.round(np.array(lbl_probs.detach().cpu().numpy()) * 100, 2)
        )
    )


# Store the trained weights
if store_weights:
    torch.save(
        best_val_layers, "data/retrained_snntorch_{}.pt".format(training_datetime)
    )
    print("*** weights stored ***")


torch.save(best_val_layers, "data/retrained_snntorch_{}.pt".format(training_datetime))
print("*** weights stored ***")
