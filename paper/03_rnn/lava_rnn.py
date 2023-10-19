import nir
import nirtorch
import torch
import numpy as np
import lava.lib.dl.slayer as slayer


SCALE = 1 << 10


def _create_rnn_subgraph(graph: nir.NIRGraph, lif_nk: str, w_nk: str) -> nir.NIRGraph:
    """Take a NIRGraph plus the node keys for a LIF and a W_rec, and return a new NIRGraph
    which has the RNN subgraph replaced with a subgraph (i.e., a single NIRGraph node).
    """
    # NOTE: assuming that the LIF and W_rec have keys of form xyz.abc
    sg_key = lif_nk.split('.')[0]  # TODO: make this more general?

    # create subgraph for RNN
    sg_edges = [
        (lif_nk, w_nk), (w_nk, lif_nk), (lif_nk, f'{sg_key}.output'), (f'{sg_key}.input', w_nk)
    ]
    sg_nodes = {
        lif_nk: graph.nodes[lif_nk],
        w_nk: graph.nodes[w_nk],
        f'{sg_key}.input': nir.Input(graph.nodes[lif_nk].input_type),
        f'{sg_key}.output': nir.Output(graph.nodes[lif_nk].output_type),
    }
    sg = nir.NIRGraph(nodes=sg_nodes, edges=sg_edges)

    # remove subgraph edges from graph
    graph.edges = [e for e in graph.edges if e not in [(lif_nk, w_nk), (w_nk, lif_nk)]]
    # remove subgraph nodes from graph
    graph.nodes = {k: v for k, v in graph.nodes.items() if k not in [lif_nk, w_nk]}

    # change edges of type (x, lif_nk) to (x, sg_key)
    graph.edges = [(e[0], sg_key) if e[1] == lif_nk else e for e in graph.edges]
    # change edges of type (lif_nk, x) to (sg_key, x)
    graph.edges = [(sg_key, e[1]) if e[0] == lif_nk else e for e in graph.edges]

    # insert subgraph into graph and return
    graph.nodes[sg_key] = sg
    return graph


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> nir.NIRGraph:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    print('replace rnn subgraph with nirgraph')

    if len(set(graph.edges)) != len(graph.edges):
        print('[WARNING] duplicate edges found, removing')
        graph.edges = list(set(graph.edges))

    # find cycle of LIF <> Dense nodes
    for edge1 in graph.edges:
        for edge2 in graph.edges:
            if not edge1 == edge2:
                if edge1[0] == edge2[1] and edge1[1] == edge2[0]:
                    lif_nk = edge1[0]
                    lif_n = graph.nodes[lif_nk]
                    w_nk = edge1[1]
                    w_n = graph.nodes[w_nk]
                    is_lif = isinstance(lif_n, (nir.LIF, nir.CubaLIF))
                    is_dense = isinstance(w_n, (nir.Affine, nir.Linear))
                    # check if the dense only connects to the LIF
                    w_out_nk = [e[1] for e in graph.edges if e[0] == w_nk]
                    w_in_nk = [e[0] for e in graph.edges if e[1] == w_nk]
                    is_rnn = len(w_out_nk) == 1 and len(w_in_nk) == 1
                    # check if we found an RNN - if so, then parse it
                    if is_rnn and is_lif and is_dense:
                        graph = _create_rnn_subgraph(graph, edge1[0], edge1[1])
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
    lif_size = int(list(input_node.input_type.values())[0][0])  # NOTE: needed for lava-dl
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
                scale=SCALE,
                # scale=4096,
            )
        )
        block.synapse.weight.data = torch.eye(7).reshape(block.synapse.weight.shape)
        return block

    elif isinstance(node, nir.NIRGraph):
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)

        if isinstance(lif_node, nir.LIF):
            raise NotImplementedError('LIF in subgraph not supported')

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
                    scale=SCALE,
                    # scale=4096,
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

    elif isinstance(node, nir.LIF):
        raise NotImplementedError('not implemented for lava-dl yet')

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

    test_data_path = "data/ds_test.pt"
    ds_test = torch.load(test_data_path)
