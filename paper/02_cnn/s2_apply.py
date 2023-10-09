# from collections import defaultdict

# import norse
import numpy as np

# import tonic
# import torch
# import tqdm
from spinnaker2 import s2_nir  # brian2_sim, hardware, helpers, s2_nir

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

# remove all batch dimensions
for k in nlist:
    input_shape = n[k].input_type["input"]
    n[k].input_type["input"] = input_shape[input_shape != 1]
    output_shape = n[k].output_type["output"]
    n[k].output_type["output"] = output_shape[output_shape != 1]

for k in nlist:
    with np.printoptions(threshold=20, edgeitems=1):
        # print((k, type(n[k]), n[k]))
        print((k, type(n[k])), n[k].input_type, n[k].output_type)

net, inp, outp = s2_nir.from_nir(
    nir_graph, outputs=["v", "spikes"], discretization_timestep=0.0001, conn_delay=0
)
