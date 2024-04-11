"""
Sharp edges:
- in lava-dl, the current and voltage state is not automatically reset. must do this manually after every forward pass.
"""
import nir
import nirtorch
import numpy as np
from dataclasses import dataclass
from functools import partial
from enum import Enum

# from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
# from lava.proc.monitor.process import Monitor
# from lava.magma.core.run_conditions import RunSteps
# from lava.proc.io.source import RingBuffer
# from lava.proc.io.sink import RingBuffer as Sink
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
import lava.lib.dl.slayer as slayer
import torch


LavaLibrary = Enum('LavaLibrary', 'Lava LavaDl')


@dataclass
class ImportConfig:
    dt: float = 1e-4
    fixed_pt: bool = True
    on_chip: bool = False
    library_preference: LavaLibrary = LavaLibrary.Lava

    def __post_init__(self):
        assert not (not self.fixed_pt and self.on_chip), "On-chip must use fixed-point"


##############################
# Lava helpers
##############################

def get_outport(lava_node):
    if isinstance(lava_node, Dense):
        return lava_node.a_out
    elif isinstance(lava_node, LIF):
        return lava_node.s_out
    else:
        raise ValueError(f"Unknown node type: {type(lava_node)}")


def get_inport(lava_node):
    if isinstance(lava_node, Dense):
        return lava_node.s_in
    elif isinstance(lava_node, LIF):
        return lava_node.a_in
    else:
        raise ValueError(f"Unknown node type: {type(lava_node)}")


def _nir_node_to_lava(node: nir.NIRNode, import_config: ImportConfig):
    """Convert a NIR node to a Lava node. May return a list of two Lava nodes, but only
    in the case of a LIF node, which is preceded by a Dense node."""

    if isinstance(node, nir.LIF):
        dt = import_config.dt
        # voltage leak: dv = dt / tau
        tau_mem = node.tau
        dv = dt / tau_mem
        vthr = node.v_threshold  # * 10
        # no current leak
        tau_syn = None  # 1/200
        du = 1.0  # no current leak
        # correction for input weights
        correction = dt / node.tau
        w = np.ones((1, 1))
        w *= correction

        if import_config.fixed_pt:
            dv = int(dv * 4095)
            du = int(du * 4095)
            vthr = int(vthr * 131071) >> 9
            w = (w * 256).astype(np.int32)

        lif = LIF(
            shape=(1,), # u=0., # v=0.,
            du=du,
            dv=dv,
            vth=vthr,
            bias_mant=0, bias_exp=0,  # no bias
            name='lif'
        )
        dense = Dense(weights=w)
        dense.a_out.connect(lif.a_in)
        return [dense, lif]

    elif isinstance(node, nir.Affine):
        w = node.weight
        assert np.allclose(node.bias, 0.), "Non-zero bias not supported by Lava"
        if import_config.fixed_pt:
            w = (w * 256).astype(np.int32)
        dense = Dense(weights=w)
        return dense

    elif isinstance(node, nir.Input):
        return None

    elif isinstance(node, nir.Output):
        return None

    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def import_from_nir_to_lava(graph: nir.NIRGraph, import_config: ImportConfig):
    """Convert a NIR graph to a Lava graph. This only creates the nodes and edges,
    not the input and output nodes.

    Args:
        graph (nir.NIRGraph): The graph to convert
        import_config (ImportConfig): The configuration for the import
    Returns:
        lava_nodes (Dict[str, (LavaNode | [LavaNode])]): The nodes in the Lava graph
        start_nodes (List[str]): The start nodes in the graph
        end_nodes (List[str]): The end nodes in the graph
    """
    dt = import_config.dt
    fixed_pt = import_config.fixed_pt

    lava_nodes = {
        k: _nir_node_to_lava(n, import_config) for k, n in graph.nodes.items()
        if not isinstance(n, (nir.Input, nir.Output))
    }
    start_nodes = []
    end_nodes = []

    for edge in graph.edges:
        src, dst = edge
        if isinstance(graph.nodes[src], nir.Input):
            start_nodes.append(dst)
        elif isinstance(graph.nodes[dst], nir.Output):
            end_nodes.append(src)
        else:
            src_node = lava_nodes[src][1] if isinstance(lava_nodes[src], list) else lava_nodes[src]
            dst_node = lava_nodes[dst][0] if isinstance(lava_nodes[dst], list) else lava_nodes[dst]
            src_port = get_outport(src_node)
            dst_port = get_inport(dst_node)
            src_port.connect(dst_port)

    return lava_nodes, start_nodes, end_nodes


##############################
# Lava-dl helpers
##############################

class Flatten(torch.nn.Module):
    def __init__(self, start_dim, end_dim):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        # make sure to not flatten the last dimension (time)
        end_dim = self.end_dim
        if self.end_dim == -1:
            end_dim -= 1
        elif self.end_dim == len(x.shape):
            end_dim = -2
        elif self.end_dim == len(x.shape) - 1:
            end_dim = -2
        # if end_dim != self.end_dim:
        #     print(f'FLATTEN: changed end_dim from {self.start_dim} to {end_dim}')
        return x.flatten(self.start_dim, end_dim)


def _create_rnn_subgraph(graph: nir.NIRGraph, lif_nk: str, w_nk: str) -> nir.NIRGraph:
    """Take a NIRGraph plus the node keys for a LIF and a W_rec, and return a new NIRGraph
    which has the RNN subgraph replaced with a subgraph (i.e., a single NIRGraph node).
    """
    # NOTE: assuming that the LIF and W_rec have keys of form xyz.abc
    sg_key = lif_nk.split(".")[0]  # TODO: make this more general?

    # create subgraph for RNN
    sg_edges = [
        (lif_nk, w_nk),
        (w_nk, lif_nk),
        (lif_nk, f"{sg_key}.output"),
        (f"{sg_key}.input", w_nk),
    ]
    sg_nodes = {
        lif_nk: graph.nodes[lif_nk],
        w_nk: graph.nodes[w_nk],
        f"{sg_key}.input": nir.Input(graph.nodes[lif_nk].input_type),
        f"{sg_key}.output": nir.Output(graph.nodes[lif_nk].output_type),
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


def _replace_rnn_subgraph_with_nirgraph(graph: nir.NIRGraph) -> tuple[nir.NIRGraph, int]:
    """Take a NIRGraph and replace any RNN subgraphs with a single NIRGraph node."""
    if len(set(graph.edges)) != len(graph.edges):
        print("[WARNING] duplicate edges found, removing")
        graph.edges = list(set(graph.edges))

    # find cycle of LIF <> Dense nodes
    n_subgraphs = 0
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
                        print("creating rnn subgraph within nirgraph")
                        graph = _create_rnn_subgraph(graph, edge1[0], edge1[1])
                        n_subgraphs += 1
    return graph, n_subgraphs


def _parse_rnn_subgraph(graph: nir.NIRGraph) -> tuple[nir.NIRNode, nir.NIRNode, int]:
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
    assert len(sub_nodes) == 4, "only 4-node RNN allowed in subgraph"
    try:
        input_node = [n for n in sub_nodes if isinstance(n, nir.Input)][0]
        output_node = [n for n in sub_nodes if isinstance(n, nir.Output)][0]
        lif_node = [n for n in sub_nodes if isinstance(n, (nir.LIF, nir.CubaLIF))][0]
        wrec_node = [n for n in sub_nodes if isinstance(n, (nir.Affine, nir.Linear))][0]
    except IndexError:
        raise ValueError("invalid RNN subgraph - could not find all required nodes")
    lif_size = int(
        list(input_node.input_type.values())[0][0]
    )  # NOTE: needed for lava-dl
    assert (
        lif_size == list(output_node.output_type.values())[0][0]
    ), "output size mismatch"
    assert lif_size == lif_node.v_threshold.size, "lif size mismatch (v_threshold)"
    assert lif_size == wrec_node.weight.shape[0], "w_rec shape mismatch"
    assert lif_size == wrec_node.weight.shape[1], "w_rec shape mismatch"

    return lif_node, wrec_node, lif_size


def _nir_node_to_lava_dl(node: nir.NIRNode, import_config: ImportConfig):
    """Convert a NIR node to a Lava-dl network."""
    scale_v_thr = 1.0
    debug_conv = False
    debug_pool = False
    debug_if = False
    debug_affine = False
    # TODO: new RNN params
    scale: int = 1 << 6
    dt = import_config.dt

    if isinstance(node, (nir.Input, nir.Output)):
        return None

    elif isinstance(node, nir.Conv2d):
        assert np.abs(node.bias).sum() == 0.0, "bias not supported in lava-dl"
        out_features = node.weight.shape[0]
        in_features = node.weight.shape[1]
        kernel_size = (node.weight.shape[2], node.weight.shape[3])
        # stride = int(node.stride[0])
        # assert node.stride[0] == node.stride[1], 'stride must be the same in both dimensions'
        if debug_conv:
            print(f"Conv2d with weights of shape {node.weight.shape}:")
            print(f"\t{in_features} in, {out_features} out, kernel {kernel_size}")
            print(
                f"\tstride {node.stride}, padding {node.padding}, dilation {node.dilation}"
            )
            print(f"\tgroups {node.groups}")
        conv_synapse_params = dict(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
        )
        conv = slayer.synapse.Conv(**conv_synapse_params)
        conv.weight.data = torch.from_numpy(node.weight.reshape(conv.weight.shape))
        return conv

    elif isinstance(node, nir.SumPool2d):
        if debug_pool:
            print(
                f"SumPool2d: kernel {node.kernel_size} pad {node.padding}, stride {node.stride}"
            )
        pool_synapse_params = dict(
            kernel_size=node.kernel_size,
            stride=node.stride,
            padding=node.padding,
            dilation=1,
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
        )
        return slayer.synapse.Pool(**pool_synapse_params)

    elif isinstance(node, nir.IF):
        assert (
            len(np.unique(node.v_threshold)) == 1
        ), "v_threshold must be the same for all neurons"
        assert (
            len(np.unique(node.r)) == 1
        ), "resistance must be the same for all neurons"
        v_thr = np.unique(node.v_threshold)[0]
        resistance = np.unique(node.r)[0]
        v_thr_eff = v_thr * resistance * scale_v_thr
        if debug_if:
            print(f"IF with v_thr={v_thr}, R={resistance} -> eff. v_thr={v_thr_eff}")
        cuba_neuron_params = dict(
            threshold=v_thr_eff,
            current_decay=1.0,
            voltage_decay=0.0,
            scale=4096,
        )
        return slayer.neuron.cuba.Neuron(**cuba_neuron_params)
        # alif_neuron_params = dict(
        #     threshold=v_thr_eff, threshold_step=0.0, scale=4096,
        #     refractory_decay=1.0,
        #     current_decay=1.0,
        #     voltage_decay=0.0,
        #     threshold_decay=0.0,
        # )
        # return slayer.neuron.alif.Neuron(**alif_neuron_params)

    elif isinstance(node, nir.Flatten):
        start_dim = node.start_dim if node.start_dim > 0 else node.start_dim + 1  # NOTE: batch
        return Flatten(start_dim, node.end_dim)

    elif isinstance(node, nir.Affine):
        assert node.bias is not None, "bias must be set for Affine layer"

        if debug_affine:
            print(f"Affine: weight shape: {node.weight.shape}")

        dense = slayer.synapse.Dense(
            in_neurons=node.weight.shape[1],
            out_neurons=node.weight.shape[0],
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
        )
        dense.weight = torch.nn.Parameter(
            data=torch.from_numpy(node.weight.reshape(dense.weight.shape)),
            requires_grad=True
        )
        dense.bias = torch.nn.Parameter(
            data=torch.from_numpy(node.bias.reshape(node.weight.shape[0])),
            requires_grad=True
        )
        return dense
    
    elif isinstance(node, nir.Linear):
        print("[WARNING] Linear layer not supported, using Dense instead")
        dense = slayer.synapse.Dense(
            in_neurons=node.weight.shape[1],
            out_neurons=node.weight.shape[0],
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
        )
        dense.weight = torch.nn.Parameter(
            data=torch.from_numpy(node.weight.reshape(dense.weight.shape)), 
            requires_grad=True
        )
        dense.bias = torch.nn.Parameter(
            data=torch.zeros_like((node.weight.shape[0],)),
            requires_grad=False
        )
        return dense

    elif isinstance(node, nir.CubaLIF):
        # TODO: figure out how to make the n_neurons dynamic
        n_neurons = 7

        # bias = node.v_leak * dt / node.tau_mem

        if not np.allclose(node.v_leak, 0):
            raise AssertionError("v_leak not supported in CubaLIF")  # not yet in lava-dl?
        if not np.allclose(node.r, node.tau_mem / dt):
            raise AssertionError("r not supported in CubaLIF")

        cur_decay = dt / node.tau_syn
        vol_decay = dt / node.tau_mem
        w_scale = node.w_in * (dt / node.tau_syn)
        vthr = node.v_threshold

        if np.unique(cur_decay).size != 1:
            raise AssertionError("CubaLIF cur_decay must be same for all neurons")
        if np.unique(vol_decay).size != 1:
            raise AssertionError("CubaLIF vol_decay must be same for all neurons")
        if np.unique(vthr).size != 1:
            raise AssertionError("CubaLIF v_thr must be same for all neurons")

        block = slayer.block.cuba.Dense(
            in_neurons=n_neurons,
            out_neurons=n_neurons,
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
            delay_shift=False,
            neuron_params=dict(
                threshold=np.unique(vthr)[0],
                current_decay=np.unique(cur_decay)[0],
                voltage_decay=np.unique(vol_decay)[0],
                shared_param=True,
                scale=scale,
            ),
        )
        # block.neuron.threshold_eps = 0.0

        weight_pre = torch.eye(n_neurons).reshape(block.synapse.weight.shape)
        if not np.allclose(w_scale, 1.0):
            # TODO: make sure that dims match up
            print(f"[warning] scaling weights according to w_in -> w_scale={w_scale[0]}")
            weight_pre = weight_pre * w_scale
        block.synapse.weight = torch.nn.Parameter(data=weight_pre, requires_grad=True)
        return block

    elif isinstance(node, nir.NIRGraph):
        # TODO: add failure exit case if the RNN is not the expected structure
        lif_node, wrec_node, lif_size = _parse_rnn_subgraph(node)

        if isinstance(lif_node, nir.LIF):
            raise NotImplementedError("LIF in subgraph not supported")

        elif isinstance(lif_node, nir.CubaLIF):
            # bias = lif_node.v_leak * dt / lif_node.tau_mem
            assert np.allclose(
                lif_node.v_leak, 0
            ), "v_leak not supported"  # not yet in lava-dl?
            assert np.allclose(
                lif_node.r, lif_node.tau_mem / dt
            ), "r not supported in CubaLIF"

            cur_decay = dt / lif_node.tau_syn
            vol_decay = dt / lif_node.tau_mem
            w_scale = lif_node.w_in * (dt / lif_node.tau_syn)
            vthr = lif_node.v_threshold

            assert (
                np.unique(cur_decay).size == 1
            ), "CubaLIF cur_decay must be same for all neurons"
            assert (
                np.unique(vol_decay).size == 1
            ), "CubaLIF vol_decay must be same for all neurons"
            assert (
                np.unique(vthr).size == 1
            ), "CubaLIF v_thr must be same for all neurons"

            rnn_block = slayer.block.cuba.Recurrent(
                in_neurons=lif_size,
                out_neurons=lif_size,
                weight_scale=1,
                weight_norm=False,
                pre_hook_fx=None,
                delay_shift=False,
                neuron_params=dict(
                    threshold=np.unique(vthr)[0],
                    current_decay=np.unique(cur_decay)[0],
                    voltage_decay=np.unique(vol_decay)[0],
                    shared_param=True,
                    scale=scale,
                ),
            )

            # rnn_block.neuron.threshold_eps = 0.0

            w_pre = torch.eye(lif_size).reshape(rnn_block.input_synapse.weight.shape)
            if not np.allclose(w_scale, 1.0):
                # TODO: make sure that dims match up
                print(f"[warning] scaling pre weights for w_in -> w_scale={w_scale[0]}")
                w_pre = w_pre * w_scale
            rnn_block.input_synapse.weight = torch.nn.Parameter(
                data=w_pre, requires_grad=True
            )

            wrec_shape = rnn_block.recurrent_synapse.weight.shape
            wrec = torch.from_numpy(wrec_node.weight).reshape(wrec_shape)
            rnn_block.recurrent_synapse.weight = torch.nn.Parameter(
                data=wrec, requires_grad=True
            )

            if isinstance(wrec_node, nir.Affine) and wrec_node.bias is not None:
                bias = torch.from_numpy(wrec_node.bias).reshape((lif_size))
                rnn_block.recurrent_synapse.bias = torch.nn.Parameter(
                    data=bias, requires_grad=True
                )

            return rnn_block

    elif isinstance(node, nir.LIF):
        raise NotImplementedError("LIF not implemented yet in lava-dl")

    else:
        raise ValueError(f"Unknown node type: {type(node)}")


class NIR2LavaDLNetwork(torch.nn.Module):
    def __init__(self, module_list, jens_order=False):
        super(NIR2LavaDLNetwork, self).__init__()
        self.blocks = torch.nn.ModuleList(module_list)

    def forward(self, spike):
        for block in self.blocks:
            if isinstance(block, torch.nn.Module):
                spike = block(spike)
            else:
                raise Exception("Unknown block type")
        return spike


def get_next_node_key(node_key, edges):
    possible_next_node_keys = [edge[1] for edge in edges if edge[0] == node_key]
    assert len(possible_next_node_keys) <= 1, "branching networks are not supported"
    if len(possible_next_node_keys) == 0:
        return None
    else:
        return possible_next_node_keys[0]


def import_from_nir_to_lava_dl(graph: nir.NIRGraph, import_config: ImportConfig, debug=False):
    # TODO (RNN addition): allow parsing of input and output nodes -> mapped to None
    dt = import_config.dt
    fixed_pt = import_config.fixed_pt

    # replace RNN subgraphs with NIRGraph nodes (i.e., subgraphs)
    graph, n_subgraphs = _replace_rnn_subgraph_with_nirgraph(graph)

    if n_subgraphs == 0:
        # no RNN subgraphs found, proceed as usual
        node_key = "input"
        visited_node_keys = [node_key]
        module_list = []
        while get_next_node_key(node_key, graph.edges) is not None:
            node_key = get_next_node_key(node_key, graph.edges)
            node = graph.nodes[node_key]
            assert node_key not in visited_node_keys, "cycling NIR graphs are not supported"
            visited_node_keys.append(node_key)
            if debug:
                print(f"node {node_key}: {type(node).__name__}")
            if node_key == "output":
                continue
            module_list.append(
                _nir_node_to_lava_dl(node, import_config)
            )
        assert len(visited_node_keys) == len(graph.nodes), "not all nodes visited"
        return NIR2LavaDLNetwork(module_list)
    
    else:
        # found RNN subgraphs, need to use NIRTorch to parse the network
        net = nirtorch.load(graph, partial(_nir_node_to_lava_dl, import_config=import_config))
        return net

##############################
# Main functions
##############################

def import_from_nir(graph: nir.NIRGraph, import_config: ImportConfig = None):
    if import_config is None:
        import_config = ImportConfig()

    if import_config.library_preference == LavaLibrary.Lava:
        return import_from_nir_to_lava(graph, import_config)
    elif import_config.library_preference == LavaLibrary.LavaDl:
        return import_from_nir_to_lava_dl(graph, import_config)
    else:
        raise ValueError(f"Unknown library preference: {import_config.library_preference}")
