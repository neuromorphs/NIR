import nengo
import numpy as np
import nir


class IF(nengo.SpikingRectifiedLinear):
    def step(self, dt, J, output, voltage):
        voltage += J * dt
        output[:] = (self.amplitude / dt) * np.where(voltage > 1, 1, 0)
        voltage[output > 0] = 0


def nir_to_nengo(n, dt=0.001):
    nengo_map = {}
    pre_map = {}
    post_map = {}

    model = nengo.Network()
    with model:
        filters = {}
        for name, obj in n.nodes.items():
            if isinstance(obj, nir.Input):
                node = nengo.Node(
                    None,
                    size_in=np.product(obj.input_type["input"]),
                    label=f"Input {name} {obj.input_type['input']}",
                )
                nengo_map[name] = node
                pre_map[name] = node
            elif isinstance(obj, nir.LIF):
                assert np.all(obj.r == 1)
                assert np.all(obj.v_leak == 0)
                N = obj.tau.flatten().shape[0]
                ens = nengo.Ensemble(
                    n_neurons=N,
                    dimensions=1,
                    label=f"LIF {name}",
                    neuron_type=nengo.LIF(
                        tau_rc=obj.tau[0],
                        tau_ref=0,
                        initial_state={"voltage": nengo.dists.Choice([0])},
                    ),
                    gain=np.ones(N) / obj.v_threshold,
                    bias=np.zeros(N),
                )
                nengo_map[name] = ens.neurons
                pre_map[name] = ens.neurons
                post_map[name] = ens.neurons
            elif isinstance(obj, nir.CubaLIF):
                assert np.all(obj.r == obj.r[0])
                R = obj.r[0]
                assert np.all(obj.v_leak == 0)
                assert np.all(obj.v_threshold == obj.v_threshold[0])
                assert np.all(obj.tau_syn == obj.tau_syn[0])
                assert np.all(obj.tau_mem == obj.tau_mem[0])

                tau_mem = obj.tau_mem[0]
                # tau_mem = -dt/np.log(1-dt/tau_mem)
                tau_syn = obj.tau_syn[0]
                tau_syn = -dt / np.log(1 - dt / tau_syn)

                N = obj.tau_mem.flatten().shape[0]
                ens = nengo.Ensemble(
                    n_neurons=N,
                    dimensions=1,
                    label=f"CubaLIF {name}",
                    neuron_type=nengo.LIF(
                        # neuron_type=nengo.RegularSpiking(nengo.LIFRate(
                        tau_ref=0,
                        tau_rc=tau_mem,
                        amplitude=dt,
                        initial_state={"voltage": nengo.dists.Choice([0])},
                    ),
                    gain=obj.w_in * R * np.ones(N) / obj.v_threshold,
                    bias=np.zeros(N),
                )
                nengo_map[name] = ens.neurons
                pre_map[name] = ens.neurons
                post_map[name] = ens.neurons
                filters[ens.neurons] = nengo.synapses.Lowpass(tau_syn)
            elif isinstance(obj, nir.IF):
                assert np.all(obj.r == 1)
                N = obj.r.flatten().shape[0]
                ens = nengo.Ensemble(
                    n_neurons=N,
                    dimensions=1,
                    label=f"IF {name}",
                    neuron_type=IF(
                        initial_state={"voltage": nengo.dists.Choice([0])}, amplitude=dt
                    ),
                    gain=np.ones(N) / obj.v_threshold.flatten() / dt,
                    bias=np.zeros(N),
                )
                nengo_map[name] = ens.neurons
                pre_map[name] = ens.neurons
                post_map[name] = ens.neurons
            elif isinstance(obj, nir.LI):
                filt = nengo.Node(
                    None,
                    size_in=obj.tau.flatten().shape[0],
                    label=f"LI {name} {obj.tau.shape}",
                )
                filters[filt] = nengo.synapses.Lowpass(obj.tau[0])
                nengo_map[name] = filt
                pre_map[name] = filt
                post_map[name] = filt
            elif isinstance(obj, nir.Affine):
                w = nengo.Node(
                    lambda t, x, obj=obj: obj.weight @ x + obj.bias,
                    size_in=obj.weight.shape[1],
                    size_out=obj.weight.shape[0],
                    label=f"Affine {name} ({obj.weight.shape[0]}x{obj.weight.shape[1]})",
                )
                nengo_map[name] = w
                pre_map[name] = w
                post_map[name] = w
            elif isinstance(obj, nir.Linear):
                w = nengo.Node(
                    lambda t, x, obj=obj: obj.weight @ x,
                    size_in=obj.weight.shape[1],
                    size_out=obj.weight.shape[0],
                    label=f"Affine {name} ({obj.weight.shape[0]}x{obj.weight.shape[1]})",
                )
                nengo_map[name] = w
                pre_map[name] = w
                post_map[name] = w
            elif isinstance(obj, nir.Output):
                nengo_map[name] = nengo.Node(
                    None,
                    size_in=np.product(obj.output_type["output"]),
                    label=f"Output {name} {obj.input_type['input']}",
                )
                post_map[name] = nengo_map[name]
            elif isinstance(obj, nir.Flatten):
                if name == "5":
                    size_in = 16 * 8 * 8
                elif name == "8":
                    size_in = 128
                else:
                    1 / 0
                node = nengo.Node(None, size_in=size_in, label=f"Flatten {name}")
                nengo_map[name] = node
                pre_map[name] = node
                post_map[name] = node
            elif isinstance(obj, nir.Conv2d):
                conv = nengo.Network(label=f"Conv2d {name}")

                if name == "0":
                    input_shape = (2, 34, 34)
                elif name == "2":
                    input_shape = (16, 16, 16)
                else:
                    input_shape = (16, 8, 8)

                pad = np.ones(input_shape)
                pad = np.pad(
                    pad,
                    [
                        (0, 0),
                        (obj.padding[0], obj.padding[0]),
                        (obj.padding[1], obj.padding[1]),
                    ],
                    "constant",
                    constant_values=0,
                )

                with conv:
                    ww = np.transpose(obj.weight, (2, 3, 1, 0))
                    c = nengo.Convolution(
                        n_filters=obj.weight.shape[0],
                        input_shape=pad.shape,
                        channels_last=False,
                        init=ww,
                        strides=obj.stride,
                        padding="valid",
                        kernel_size=(obj.weight.shape[2], obj.weight.shape[3]),
                    )
                    conv.input = nengo.Node(
                        None, size_in=np.product(input_shape), label=f"{name}.in"
                    )
                    conv.pad = nengo.Node(None, size_in=np.product(pad.shape))
                    nengo.Connection(
                        conv.input,
                        conv.pad[np.where(pad.flatten() > 0)[0]],
                        synapse=None,
                    )
                    conv.output = nengo.Node(
                        None, size_in=c.size_out, label=f"{name}.out"
                    )

                    nengo.Connection(conv.pad, conv.output, synapse=None, transform=c)

                nengo_map[name] = conv
                pre_map[name] = conv.output
                post_map[name] = conv.input
            elif isinstance(obj, nir.SumPool2d):
                pool = nengo.Network(label=f"SumPool2d {name}")
                with pool:
                    if name == "4":
                        input_shape = (16, 16, 16)
                    elif name == "7":
                        input_shape = (8, 8, 8)
                    else:
                        1 / 0

                    n_filters = input_shape[0]
                    pool_size = tuple(obj.kernel_size)
                    n_pool = np.product(pool_size)
                    kernel = np.reshape(
                        [np.eye(n_filters)] * n_pool, pool_size + (n_filters, n_filters)
                    )

                    c = nengo.Convolution(
                        n_filters=input_shape[0],
                        input_shape=input_shape,
                        channels_last=False,
                        init=kernel,
                        strides=obj.stride,
                        padding="valid",
                        kernel_size=pool_size,
                    )
                    pool.input = nengo.Node(
                        None,
                        size_in=np.product(input_shape),
                        label=f"SumPool2d {name}.in",
                    )
                    pool.output = nengo.Node(
                        None, size_in=c.size_out, label=f"SumPool2d {name}.out"
                    )
                    nengo.Connection(pool.input, pool.output, synapse=None, transform=c)
                nengo_map[name] = pool
                pre_map[name] = pool.output
                post_map[name] = pool.input
            else:
                raise Exception(f"Unknown object: {obj}")
        for pre, post in n.edges:
            synapse = filters.get(nengo_map[post], None)

            if pre_map[pre].size_out != post_map[post].size_in:
                print("Error")
                print("pre", pre, pre_map[pre], pre_map[pre].size_out)
                print("post", post, post_map[post], post_map[post].size_in)
                1 / 0

            else:
                nengo.Connection(pre_map[pre], post_map[post], synapse=synapse)

    return model, nengo_map
