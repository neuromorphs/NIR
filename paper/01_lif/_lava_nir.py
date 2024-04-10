import nir
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import namedtuple
# from typing import Optional
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense


LavaLibrary = Enum('LavaLibrary', 'Lava LavaDl')


# def import_on_chip_lava():
#     from lava.proc.io.sink import RingBuffer as Sink
#     from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter


@dataclass
class ImportConfig:
    dt: float = 1e-4
    fixed_pt: bool = True
    on_chip: bool = False
    library_preference: LavaLibrary = LavaLibrary.Lava

    def __post_init__(self):
        assert not (not self.fixed_pt and self.on_chip), "On-chip must use fixed-point"


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


def import_from_nir_to_lava_dl(graph: nir.NIRGraph, import_config: ImportConfig):
    dt = import_config.dt
    fixed_pt = import_config.fixed_pt
    raise NotImplementedError("Lava-dl not implemented yet")


def import_from_nir(graph: nir.NIRGraph, import_config: ImportConfig = None):
    if import_config is None:
        import_config = ImportConfig()

    if import_config.library_preference == LavaLibrary.Lava:
        return import_from_nir_to_lava(graph, import_config)
    elif import_config.library_preference == LavaLibrary.Lava_dl:
        return import_from_nir_to_lava_dl(graph, import_config)
    else:
        raise ValueError(f"Unknown library preference: {import_config.library_preference}")
