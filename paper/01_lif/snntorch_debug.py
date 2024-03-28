import snntorch as snn
import numpy as np
import torch
import matplotlib.pyplot as plt
from snntorch import spikeplot as splt


V_THR = 2.0  # TODO: STRANGE BEHAVIOR FOR *V_THR = 2.5*


class TestNet(torch.nn.Module):
    def __init__(self, reset_after=False) -> None:
        super().__init__()
        self.lif = snn.Synaptic(
            alpha=0.1,
            beta=0.96,
            threshold=V_THR,
            reset_mechanism="zero",
            init_hidden=False,
            reset_after=reset_after,
        )

    def forward(self, x):
        syn, mem = self.lif.init_synaptic()
        arr_syn, arr_mem, arr_spk = [], [], []
        for step in range(x.shape[0]):
            spk, syn, mem = self.lif(x[step], syn, mem)
            arr_syn.append(syn)
            arr_mem.append(mem)
            arr_spk.append(spk)
        return (
            torch.stack(arr_spk, dim=0),
            torch.stack(arr_syn, dim=0),
            torch.stack(arr_mem, dim=0),
        )


isis = np.array([5, 4, 3, 2, 1, 0])
spk_times = []
for isi in isis:
    spk_times += ([1] + [0] * isi * 2) * 4
spk_times = np.array(spk_times + [1] * 5 + [0] * 5)

inp = torch.from_numpy(spk_times).float()

net = TestNet(reset_after=False)
spk, syn, mem = net(inp)
net2 = TestNet(reset_after=True)
spk2, syn2, mem2 = net2(inp)

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 5))
axs[0].eventplot(np.where(spk_times == 1)[0], linelengths=0.5)
axs[0].set_ylabel("input")
axs[1].eventplot(np.where(spk.detach().numpy() == 1)[0])
axs[1].eventplot(
    np.where(spk2.detach().numpy() == 1)[0], lineoffsets=-0.5, color="orange"
)
axs[1].set_ylabel("spikes")
axs[2].plot(syn.detach().numpy())
axs[2].plot(syn2.detach().numpy() - 0.03)
axs[2].set_ylabel("current")
axs[3].plot(mem.detach().numpy())
axs[3].plot(mem2.detach().numpy() - 0.1)
axs[3].hlines(V_THR, 0, mem.shape[0], color="r", ls="--")
axs[3].set_ylabel("membrane")
plt.show()
