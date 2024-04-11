import nir
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import namedtuple

# from lava.proc.monitor.process import Monitor
# from lava.magma.core.run_conditions import RunSteps
# from lava.proc.io.source import RingBuffer
# from lava.proc.lif.process import LIF
# from lava.proc.dense.process import Dense
# from lava.proc.io.sink import RingBuffer as Sink
# from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
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


def handle_imports(import_config: ImportConfig):
    if import_config.library_preference == LavaLibrary.Lava:
        from lava.proc.monitor.process import Monitor
        from lava.magma.core.run_conditions import RunSteps
        from lava.proc.io.source import RingBuffer
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        if import_config.on_chip:
            from lava.proc.io.sink import RingBuffer as Sink
            from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
    if import_config.library_preference == LavaLibrary.LavaDl:
        import lava.lib.dl.slayer as slayer
        import torch

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


def _nir_node_to_lava_dl(node: nir.NIRNode, import_config: ImportConfig):
    """Convert a NIR node to a Lava-dl network."""
    scale_v_thr = 1.0
    debug_conv = False
    debug_pool = False
    debug_if = False
    debug_affine = False

    if isinstance(node, nir.ir.Conv2d):
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

    elif isinstance(node, nir.ir.SumPool2d):
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

    elif isinstance(node, nir.ir.IF):
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

    elif isinstance(node, nir.ir.Affine):
        assert np.abs(node.bias).sum() == 0.0, "bias not supported in lava-dl"
        weight = node.weight
        out_neurons = weight.shape[0]
        in_neurons = weight.shape[1]
        if debug_affine:
            print(f"Affine: weight shape: {weight.shape}")
        dense = slayer.synapse.Dense(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            weight_scale=1,
            weight_norm=False,
            pre_hook_fx=None,
        )
        dense.weight.data = torch.from_numpy(weight.reshape(dense.weight.shape))
        return dense

    elif isinstance(node, nir.ir.Flatten):
        start_dim = node.start_dim if node.start_dim > 0 else node.start_dim + 1  # NOTE: batch
        return Flatten(start_dim, node.end_dim)

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
    dt = import_config.dt
    fixed_pt = import_config.fixed_pt
    # raise NotImplementedError("Lava-dl not implemented yet")
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

##############################
# Main functions
##############################

def import_from_nir(graph: nir.NIRGraph, import_config: ImportConfig = None):
    if import_config is None:
        import_config = ImportConfig()

    handle_imports(import_config)

    if import_config.library_preference == LavaLibrary.Lava:
        return import_from_nir_to_lava(graph, import_config)
    elif import_config.library_preference == LavaLibrary.LavaDl:
        return import_from_nir_to_lava_dl(graph, import_config)
    else:
        raise ValueError(f"Unknown library preference: {import_config.library_preference}")
