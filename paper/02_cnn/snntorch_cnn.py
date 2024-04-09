import tonic
import torch
import snntorch as snn
import numpy as np
from snntorch import import_nirtorch
import nir
import tqdm


graph = nir.read("cnn_sinabs.nir")
graph.nodes.keys()

net = import_nirtorch.from_nir(graph)
print(net)

inp_data = torch.from_numpy(np.load("cnn_numbers.npy")).float()
print('input data:', inp_data.shape)
modules = [e.elem for e in net.get_execution_order()]

# init all I&F neurons
mem_dict = {}
for idx, module in enumerate(modules):
    if isinstance(module, snn.Leaky):
        mem_dict[idx] = module.init_leaky()
# forward pass through time
out = []
arr_spk_layer = []
for t in range(inp_data.shape[0]):
    x = inp_data[t]
    spklayer = None
    for idx, module in enumerate(modules):
        if isinstance(module, snn.Leaky):
            x, mem_dict[idx] = module(x, mem_dict[idx])
            if spklayer is None:
                spklayer = x.detach()
        else:
            x = module(x)
    out.append(x)
    arr_spk_layer.append(spklayer)
arr_spk_layer = torch.stack(arr_spk_layer).detach()
# out = torch.stack(out).detach()
np.save("snnTorch_activity.npy", arr_spk_layer.numpy())

############################################################
# sample_idx = 9020
# inp = test_ds[sample_idx][0]
# lbl = test_ds[sample_idx][1]
############################################################

bs = 128
collate = tonic.collation.PadTensors(batch_first=False)
to_frame = tonic.transforms.ToFrame(
    sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1e3
)
test_ds = tonic.datasets.NMNIST("./nmnist", transform=to_frame, train=False)
test_dl = torch.utils.data.DataLoader(
    test_ds, shuffle=True, batch_size=bs, collate_fn=collate
)

accuracies = []
pbar = tqdm.tqdm(total=len(test_dl), desc="Processing", position=0, leave=True)
for idx, (x, y) in enumerate(test_dl):
    # x = torch.moveaxis(x, 0, -1)

    # reset/init I&F neurons
    mem_dict = {}
    for idx, module in enumerate(modules):
        if isinstance(module, snn.Leaky):
            mem_dict[idx] = module.init_leaky()

    # forward pass through time
    out = []
    for t in range(x.shape[0]):
        xt = x[t]
        for idx, module in enumerate(modules):
            if isinstance(module, snn.Leaky):
                xt, mem_dict[idx] = module(xt, mem_dict[idx])
            else:
                xt = module(xt)
        out.append(xt)
    out = torch.stack(out).detach()

    pred = out.mean(0).argmax(-1)
    accuracy = (pred == y).sum() / x.shape[1]
    accuracies.append(accuracy)
    pbar.set_postfix(accuracy="{:.2f}%".format(sum(accuracies) / len(accuracies) * 100))
    pbar.update(1)
pbar.close()
accuracies = np.array(accuracies)
print(f"accuracy: {accuracies.mean():.2%} +/- {accuracies.std():.2%}")
np.save("snntorch_accuracies.npy", accuracies)
np.save("snntorch_accuracy.npy", accuracies.mean())
