import nengo
import numpy as np

import nir

n = nir.NIRGraph(
    nodes=[
        nir.Input(shape=np.array([3])),
        nir.Affine(weight=np.array([[8, 2, 10], [14, 3, 14]]).T, bias=np.array([1, 2])),
        nir.LIF(
            tau=np.array([1] * 2),
            r=np.array([1] * 2),
            v_leak=np.array([0] * 2),
            v_threshold=np.array([1] * 2),
        ),
        nir.Output(shape=np.array([3])),
    ],
    edges=[(0, 1), (1, 2), (2, 3)],
)


def nir_to_nengo(n, swap_linear_order=False):
    nengo_map = []

    model = nengo.Network()
    with model:
        filters = {}
        for i, obj in enumerate(n.nodes):
            if isinstance(obj, nir.Input):
                node = nengo.Node(np.zeros(obj.shape), label=f"Input {i} {obj.shape}")
                nengo_map.append(node)
            elif isinstance(obj, nir.LIF):
                N = obj.tau.flatten().shape[0]
                ens = nengo.Ensemble(
                    n_neurons=N,
                    dimensions=1,
                    label=f"LIF {i}",
                    neuron_type=nengo.RegularSpiking(
                        nengo.LIFRate(tau_rc=obj.tau[0], tau_ref=0)
                    ),
                    # neuron_type=nengo.LIF(tau_rc=obj.tau[0], tau_ref=0),
                    gain=np.ones(N),
                    bias=np.zeros(N),
                )
                nengo_map.append(ens.neurons)
            elif isinstance(obj, nir.LI):
                filt = nengo.Node(
                    lambda t, x: x,
                    size_in=obj.tau.flatten().shape[0],
                    label=f"LI {i} {obj.tau.shape}",
                )
                filters[filt] = nengo.synapses.Lowpass(obj.tau[0])
                nengo_map.append(filt)
            elif isinstance(obj, nir.Affine):
                weights = obj.weight
                if swap_linear_order:
                    weights = weights.T
                w = nengo.Node(
                    lambda t, x, obj=obj: weights @ x + obj.bias,
                    size_in=weights.shape[1],
                    size_out=weights.shape[0],
                    label=f"({weights.shape[0]}x{weights.shape[1]})",
                )
                nengo_map.append(w)
            elif isinstance(obj, nir.Output):
                nengo_map.append(
                    None
                )  # because NIR spec doesn't tell me the size, I can't create this yet
            else:
                raise Exception(f"Unknown NIR object: {obj}")
        for pre, post in n.edges:
            if nengo_map[post] is None:
                output = nengo.Node(
                    lambda t, x: x,
                    size_in=nengo_map[pre].size_out,
                    label=f"Output {post}",
                )
                nengo_map[post] = output
            synapse = filters.get(nengo_map[post], None)

            if nengo_map[pre].size_out != nengo_map[post].size_in:
                print("Error")
                print("pre", nengo_map[pre])
                print("post", nengo_map[post])
                1 / 0

            else:
                nengo.Connection(nengo_map[pre], nengo_map[post], synapse=synapse)

    return model


model = nir_to_nengo(n, swap_linear_order=True)