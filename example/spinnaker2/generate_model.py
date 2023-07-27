import numpy as np
from spinnaker2 import s2_nir

import nir

nir_model = nir.NIRGraph.from_list(
    nir.Input(shape=np.array([3])),
    nir.Affine(
        weight=np.array([[8, 2, 10], [14, 3, 14]]).T * 32, bias=np.array([0, 8]) * 32
    ),
    nir.LIF(
        tau=np.array([4] * 2),
        r=np.array([1.25, 0.8]),
        v_leak=np.array([0.5] * 2),
        v_threshold=np.array([5] * 2) * 32,
    ),
    nir.Output(shape=np.array([2])),
)
nir.write("nir_model.hdf5", nir_model)

print(nir_model)
print("read back")
net = s2_nir.from_nir(nir_model)
print(net)
