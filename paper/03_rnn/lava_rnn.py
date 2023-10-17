import nir
import nirtorch
import torch
import numpy as np
import lava.lib.dl.slayer as slayer


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> nir.NIRGraph:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    if len([e for e in graph.nodes.values() if isinstance(e, nir.Input)]) > 1:
        print('found RNN subgraph, trying to parse')
        cand_sg_nk = list(set([e[1] for e in graph.edges if e[1] not in graph.nodes]))
        print('detected subgraph! candidates:', cand_sg_nk)
        assert len(cand_sg_nk) == 1, 'only one subgraph allowed'
        nk = cand_sg_nk[0]
        nodes = {k: v for k, v in graph.nodes.items() if k.startswith(f'{nk}.')}
        edges = [e for e in graph.edges if e[0].startswith(f'{nk}.') or e[1].startswith(f'{nk}.')]
        valid_edges = all([e[0].startswith(f'{nk}.') for e in edges])
        valid_edges = valid_edges and all([e[1].startswith(f'{nk}.') for e in edges])
        assert valid_edges, 'subgraph edges must start with subgraph key'
        sg_graph = nir.NIRGraph(nodes=nodes, edges=edges)
        for k in nodes.keys():
            graph.nodes.pop(k)
        for e in edges:
            graph.edges.remove(e)
        graph.nodes[nk] = sg_graph
    return graph


def _parse_rnn_subgraph(graph: nir.NIRGraph) -> (nir.NIRNode, nir.NIRNode, int):
    """Try parsing the graph as a RNN subgraph.

    Assumes four nodes: Input, Output, LIF | CubaLIF, Affine | Linear
    Checks that all nodes have consistent shapes.
    Will throw an error if either not all nodes are found or consistent shapes are found.

    Returns:
        lif_node: LIF | CubaLIF node
        wrec_node: Affine | Linear node
        lif_size: int, number of neurons in the RNN
    """
    sub_nodes = graph.nodes.values()
    assert len(sub_nodes) == 4, 'only 4-node RNN allowed in subgraph'
    try:
        input_node = [n for n in sub_nodes if isinstance(n, nir.Input)][0]
        output_node = [n for n in sub_nodes if isinstance(n, nir.Output)][0]
        lif_node = [n for n in sub_nodes if isinstance(n, (nir.LIF, nir.CubaLIF))][0]
        wrec_node = [n for n in sub_nodes if isinstance(n, (nir.Affine, nir.Linear))][0]
    except IndexError:
        raise ValueError('invalid RNN subgraph - could not find all required nodes')
    lif_size = int(list(input_node.input_type.values())[0][0])
    assert lif_size == list(output_node.output_type.values())[0][0], 'output size mismatch'
    assert lif_size == lif_node.v_threshold.size, 'lif size mismatch (v_threshold)'
    assert lif_size == wrec_node.weight.shape[0], 'w_rec shape mismatch'
    assert lif_size == wrec_node.weight.shape[1], 'w_rec shape mismatch'

    return lif_node, wrec_node, lif_size


def _nir_to_lavadl_module(node: nir.NIRNode, hack_w_scale=True) -> torch.nn.Module:
    if isinstance(node, nir.Input) or isinstance(node, nir.Output):
        return None

    elif isinstance(node, nir.Affine):
        assert node.bias is not None, 'bias must be specified for Affine layer'

        mod = slayer.synapse.Dense(
            in_neurons=node.weight.shape[1], out_neurons=node.weight.shape[0],
            weight_scale=1, weight_norm=False, pre_hook_fx=None
        )
        mod.weight.data = torch.from_numpy(node.weight.reshape(mod.weight.shape))
        if not np.allclose(node.bias, 0.):
            bias = torch.from_numpy(node.bias.reshape((node.weight.shape[0])))
            mod.bias = torch.nn.Parameter(data=bias, requires_grad=True)
        return mod

    elif isinstance(node, nir.Linear):
        mod = slayer.synapse.Dense(
            in_neurons=node.weight.shape[1], out_neurons=node.weight.shape[0],
            weight_scale=1, weight_norm=False, pre_hook_fx=None
        )
        mod.weight.data = torch.from_numpy(node.weight.reshape(mod.weight.shape))
        return mod

    elif isinstance(node, nir.LIF):
        raise NotImplementedError('not implemented for lava-dl yet')
        dt = 1e-4

        assert np.allclose(node.v_leak, 0.), 'v_leak not supported'
        assert np.unique(node.v_threshold).size == 1, 'v_threshold must be same for all neurons'

        beta = 1 - (dt / node.tau)
        vthr = node.v_threshold
        w_scale = node.r * dt / node.tau

        if not np.allclose(w_scale, 1.):
            if hack_w_scale:
                vthr = vthr / np.unique(w_scale)[0]
                print('[warning] scaling weights to avoid scaling inputs')
                print(f'w_scale: {w_scale}, r: {node.r}, dt: {dt}, tau: {node.tau}')
            else:
                raise NotImplementedError('w_scale must be 1, or the same for all neurons')

        assert np.unique(vthr).size == 1, 'LIF v_thr must be same for all neurons'

        return snn.Leaky(
            beta=beta,
            threshold=np.unique(vthr)[0],
            reset_mechanism='zero',
            init_hidden=True,
        )

    elif isinstance(node, nir.CubaLIF):
        dt = 1e-4

        assert np.allclose(node.v_leak, 0), 'v_leak not supported'  # not yet in lava-dl?
        assert np.allclose(node.r, node.tau_mem / dt), 'r not supported in CubaLIF'

        cur_decay = dt / node.tau_syn
        vol_decay = dt / node.tau_mem
        # bias = node.v_leak * dt / node.tau_mem
        w_scale = node.w_in * (dt / node.tau_syn)
        vthr = node.v_threshold

        if not np.allclose(w_scale, 1.):
            if hack_w_scale:
                vthr = vthr / w_scale
                print(f'[warning] scaling weights to avoid scaling inputs, w_scale: {w_scale[0]}')
                print(f'w_in: {node.w_in[0]}, dt: {dt}, tau_syn: {node.tau_syn[0]}')
            else:
                raise NotImplementedError('w_scale must be 1, or the same for all neurons')

        assert np.unique(cur_decay).size == 1, 'CubaLIF cur_decay must be same for all neurons'
        assert np.unique(vol_decay).size == 1, 'CubaLIF vol_decay must be same for all neurons'
        assert np.unique(vthr).size == 1, 'CubaLIF v_thr must be same for all neurons'

        block = slayer.block.cuba.Dense(
            in_neurons=7,  # HACK: hard-coded
            out_neurons=7,  # HACK: hard-coded
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
            neuron_params=dict(
                threshold=np.unique(vthr)[0],
                current_decay=np.unique(cur_decay)[0],
                voltage_decay=np.unique(vol_decay)[0],
                shared_param=False,
                scale=4096,
            )
        )
        block.synapse.weight.data = torch.eye(7).reshape(block.synapse.weight.shape)
        return block

        return slayer.neuron.cuba.Neuron(
            threshold=np.unique(vthr)[0],
            current_decay=np.unique(cur_decay)[0],
            voltage_decay=np.unique(vol_decay)[0],
            # bias=bias,
            shared_param=False,
            scale=4096,
        )
        # alpha = 1 - (dt / node.tau_syn)
        # beta = 1 - (dt / node.tau_mem)
        # vthr = node.v_threshold
        # w_scale = node.w_in * (dt / node.tau_syn)
        # return snn.Synaptic(
        #     alpha=alpha,
        #     beta=beta,
        #     threshold=np.unique(vthr)[0],
        #     reset_mechanism='zero',
        #     init_hidden=True,
        # )

    elif isinstance(node, nir.NIRGraph):
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)

        if isinstance(lif_node, nir.LIF):
            raise NotImplementedError('LIF in subgraph not supported')
            # TODO: fix neuron parameters
            rleaky = snn.RLeaky(
                beta=1 - (1 / lif_node.tau),
                threshold=lif_node.v_threshold,
                reset_mechanism='zero',
                init_hidden=True,
                all_to_all=True,
                linear_features=lif_size,
            )
            rleaky.recurrent.weight.data = torch.Tensor(wrec_node.weight)
            if isinstance(wrec_node, nir.Affine):
                rleaky.recurrent.bias.data = torch.Tensor(wrec_node.bias)
            return rleaky

        elif isinstance(lif_node, nir.CubaLIF):
            dt = 1e-4

            assert np.allclose(lif_node.v_leak, 0), 'v_leak not supported'  # not yet in lava-dl?
            assert np.allclose(lif_node.r, lif_node.tau_mem / dt), 'r not supported in CubaLIF'

            cur_decay = dt / lif_node.tau_syn
            vol_decay = dt / lif_node.tau_mem
            # bias = lif_node.v_leak * dt / lif_node.tau_mem
            w_scale = lif_node.w_in * (dt / lif_node.tau_syn)
            vthr = lif_node.v_threshold

            if not np.allclose(w_scale, 1.):
                if hack_w_scale:
                    vthr = vthr / w_scale
                    print(f'[warning] scaling weights to avoid scaling inputs, w_scale: {w_scale[0]}')
                    print(f'w_in: {lif_node.w_in[0]}, dt: {dt}, tau_syn: {lif_node.tau_syn[0]}')
                else:
                    raise NotImplementedError('w_scale must be 1, or the same for all neurons')

            assert np.unique(cur_decay).size == 1, 'CubaLIF cur_decay must be same for all neurons'
            assert np.unique(vol_decay).size == 1, 'CubaLIF vol_decay must be same for all neurons'
            assert np.unique(vthr).size == 1, 'CubaLIF v_thr must be same for all neurons'

            rnn_block = slayer.block.cuba.Recurrent(
                in_neurons=lif_size,
                out_neurons=lif_size,
                weight_scale=1,
                weight_norm=False,
                pre_hook_fx=None,
                neuron_params=dict(
                    threshold=np.unique(vthr)[0],
                    current_decay=np.unique(cur_decay)[0],
                    voltage_decay=np.unique(vol_decay)[0],
                    shared_param=False,
                    scale=4096,
                )
            )

            in_shape = rnn_block.input_synapse.weight.shape
            rnn_block.input_synapse.weight.data = torch.eye(lif_size).reshape(in_shape)
            wrec_shape = rnn_block.recurrent_synapse.weight.shape
            wrec = torch.from_numpy(wrec_node.weight)
            rnn_block.recurrent_synapse.weight.data = wrec.reshape(wrec_shape)

            if isinstance(wrec_node, nir.Affine) and wrec_node.bias is not None:
                bias = torch.from_numpy(wrec_node.bias)
                rnn_block.recurrent_synapse.bias = torch.nn.Parameter(bias.reshape((lif_size)))

            return rnn_block

            # breakpoint()

            # rsynaptic = snn.RSynaptic(
            #     alpha=alpha,
            #     beta=beta,
            #     threshold=np.unique(vthr)[0],
            #     reset_mechanism='zero',
            #     init_hidden=True,
            #     all_to_all=not diagonal,
            #     linear_features=lif_size,
            #     V=np.diag(wrec_node.weight) if diagonal else None,
            # )

            # rsynaptic.recurrent.weight.data = torch.Tensor(wrec_node.weight)
            # if isinstance(wrec_node, nir.Affine):
            #     rsynaptic.recurrent.bias.data = torch.Tensor(wrec_node.bias)
            # return rsynaptic

    else:
        print('[WARNING] could not parse node of type:', node.__class__.__name__)

    return None


def from_nir(graph: nir.NIRGraph) -> torch.nn.Module:
    # find valid RNN subgraphs, and replace them with a single NIRGraph node
    graph = _replace_rnn_subgraph_with_nirgraph(graph)
    # TODO: right now, the subgraph edges seem to not be parsed correctly - fix this
    return nirtorch.load(graph, _nir_to_lavadl_module)


if __name__ == '__main__':
    nirgraph = nir.read('braille_v2.nir')
    net = from_nir(nirgraph)
