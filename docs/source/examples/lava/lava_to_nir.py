import typing
import h5py
import re
import pathlib
import nir
import numpy as np
import logging
from typing import Tuple
import collections.abc


PATH_TYPE = typing.Union[str, pathlib.Path]


def read_cuba_lif(layer: h5py.Group, shape: Tuple[int] = None) -> nir.NIRNode:
    """Reads a CUBA LIF layer from a h5py.Group.
    TODOs: 
    - what if the layer is more than 1D?
    - handle scaleRho tauRho theta
    - support graded spikes
    - support refdelay
    - support other neuron types
    If the neuron model is not supported, a warning is logged and None is returned.
    """
    logging.debug(f"read_cuba_lif {layer['neuron']['type'][()]}")

    if 'gradedSpike' in layer['neuron']:
        if layer['neuron']['gradedSpike'][()]:
            logging.warning('graded spikes not supported')

    if layer['neuron']['type'][()] in [b'LOIHI', b'CUBA']:
        if layer['neuron']['refDelay'][()] != 1:
            logging.warning('refdelay not supported, setting to 1')
        if layer['neuron']['vDecay'] == 0:
            logging.warning('vDecay is 0, setting to inf')
        if layer['neuron']['iDecay'] == 0:
            logging.warning('iDecay is 0, setting to inf')

        # Lava-dl exports hardware fixed tipe to hdf5. For NIR, export in floating point. 
        dt      = 1e-4  # This dt value is according to nir_to_lava.py script. https://github.com/neuromorphs/NIR/blob/main/paper/nir_to_lava.py#L67
        vdecay  = layer['neuron']['vDecay'][()]  / 4096 # Save the value in NIR as floating point
        idecay  = layer['neuron']['iDecay'][()]  / 4096 # Save the value in NIR as floating point
        thr     = layer['neuron']['vThMant'][()] / 64   # Save the value in NIR as floating point
        tau_mem = dt/float(vdecay) if vdecay != 0 else np.inf
        tau_syn = dt/float(idecay) if idecay != 0 else np.inf
        shape   = layer['weight'].shape[0]
        r       = tau_mem/dt # no scaling of synaptic current
        w_in    = tau_syn/dt # no scaling of synaptic voltage

        return nir.CubaLIF(
            tau_syn=np.full(shape, tau_syn),
            tau_mem=np.full(shape, tau_mem),
            r=np.full(shape, r),  
            v_leak=np.full(shape, 0.),  # currently no bias in Loihi's neurons
            v_threshold=np.full(shape, thr),
            w_in=np.full(shape,w_in),  
            v_reset=np.full(shape,0.) # LAVA-DL CUBA LiF always reset to 0
        )
    else:
        logging.warning('currently only support for CUBA-LIF')
        logging.error(f"no support for {layer['neuron']['type'][()]}")
        return None


def read_node(network: h5py.Group) -> nir.NIRNode:
    """Read a graph from a HDF/conn5 file.
    
    TODOs:
    - support delay in convolutional layers
    """
    nodes = []
    edges = []
    current_shape = None

    # need to sort keys as integers, otherwise does 1->10->2
    layer_keys = sorted(list(map(int, network.keys())))

    # iterate over layers
    for layer_idx_int in layer_keys:
        layer_idx = str(layer_idx_int)
        layer = network[layer_idx]

        logging.info(f"--- Layer #{layer_idx}: {layer['type'][0].decode().upper()}")
        logging.debug(f'current shape: {current_shape}')

        if layer['type'][0] == b'dense':
            # shape, type, neuron, inFeatures, outFeatures, weight, delay?
            logging.debug(f'dense weights of shape {layer["weight"][:].shape}')

            # make sure weight matrix matches shape of previous layer
            if current_shape is None:
                assert len(layer['weight'][:].shape) == 2, 'shape mismatch in dense'
                current_shape = layer['weight'][:].shape[1]
            elif isinstance(current_shape, int):
                assert current_shape == layer['weight'][:].shape[-1], 'shape mismatch in dense'
            else:
                assert len(current_shape) == 1, 'shape mismatch in dense'
                assert current_shape[0] == layer['weight'][:].shape[1], 'shape mismatch in dense'

            # infer shape of current layer
            assert len(layer['weight'][:].shape) in [1, 2], 'invalid dimension for dense layer'
            current_shape = 1 if len(layer['weight'][:].shape) == 1 else layer['weight'][:].shape[0]

            # store the weight matrix (np.array, carrying over type)
            if 'bias' in layer:
                nodes.append(nir.Affine(weight=(layer['weight'][:] / 64), bias=layer['bias'][:]))
            else:
                nodes.append(nir.Linear(weight=(layer['weight'][:] / 64))) # Save weights as floating point as well.

            # store the neuron group
            neuron = read_cuba_lif(layer)
            if neuron is None:
                raise NotImplementedError('could not read neuron')
            nodes.append(neuron)

            # connect linear to neuron, neuron to next element
            edges.append((len(nodes)-2, len(nodes)-1))
            edges.append((len(nodes)-1, len(nodes)))

        elif layer['type'][0] == b'input':
            # iDecay, refDelay, scaleRho, tauRho, theta, type, vDecay, vThMant, wgtExp
            current_shape = layer['shape'][:]
            logging.warning('INPUT - not implemented yet')
            logging.debug(f'keys: {layer.keys()}')
            logging.debug(f'shape: {layer["shape"][:]}, bias: {layer["bias"][()]}, weight: {layer["weight"][()]}')
            logging.debug(f'neuron keys: {", ".join(list(layer["neuron"].keys()))}')

        elif layer['type'][0] == b'flatten':
            # shape, type
            logging.debug(f"flattening shape (ignored): {layer['shape'][:]}")
            # check last layer's size
            assert len(nodes) > 0, 'flatten layer: must be preceded by a layer'
            assert isinstance(current_shape, tuple), 'flatten layer: nothing to flatten'
            last_node = nodes[-1]
            nodes.append(nir.Flatten(n_dims=1))
            current_shape = int(np.prod(current_shape))
            edges.append((len(nodes)-1, len(nodes)))

        elif layer['type'][0] == b'conv':
            # shape, type, neuron, inChannels, outChannels, kernelSize, stride, 
            # padding, dilation, groups, weight, delay?
            weight = layer['weight'][:]
            stride = layer['stride'][()]
            pad = layer['padding'][()]
            dil = layer['dilation'][()]
            kernel_size = layer['kernelSize'][()]
            in_channels = layer['inChannels'][()]
            out_channels = layer['outChannels'][()]
            logging.debug(f'stride {stride} padding {pad} dilation {dil} w {weight.shape}')

            # infer shape of current layer
            assert in_channels == current_shape[0], 'in_channels must match previous layer'
            x_prev = current_shape[1]
            y_prev = current_shape[2]
            x = (x_prev + 2*pad[0] - dil[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
            y = (y_prev + 2*pad[1] - dil[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
            current_shape = (out_channels, x, y)

            # check for unsupported options
            if layer['groups'][()] != 1:
                logging.warning('groups not supported, setting to 1')
            if 'delay' in layer:
                logging.warning(f"delay=({layer['delay'][()]}) not supported, ignoring")

            # store the conv matrix (np.array, carrying over type)
            nodes.append(nir.Conv2d(
                weight=layer['weight'][:],
                bias=layer['bias'][:] if 'bias' in layer else None,
                stride=stride,
                padding=pad,
                dilation=dil,
                groups=layer['groups'][()]
            ))

            # store the neuron group
            neuron = read_cuba_lif(layer)
            if neuron is None:
                raise NotImplementedError('could not read neuron')
            nodes.append(neuron)

            # connect conv to neuron group, neuron group to next element
            edges.append((len(nodes)-2, len(nodes)-1))
            edges.append((len(nodes)-1, len(nodes)))

        elif layer['type'][0] == b'average':
            # shape, type
            logging.error('AVERAGE LAYER - not implemented yet')
            raise NotImplementedError('average layer not implemented yet')

        elif layer['type'][0] == b'concat':
            # shape, type, layers
            logging.error('CONCAT LAYER - not implemented yet')
            raise NotImplementedError('concat layer not implemented yet')

        elif layer['type'][0] == b'pool':
            # shape, type, neuron, kernelSize, stride, padding, dilation, weight
            logging.error('POOL LAYER - not implemented yet')
            raise NotImplementedError('pool layer not implemented yet')

        else:
            logging.error('layer type not supported:', layer['type'][0])

    # remove last edge (no next element)
    edges.pop(-1)

    return nir.NIRGraph(nodes={i: node for i, node in enumerate(nodes)}, edges=edges)


def convert_to_nir(net_config: PATH_TYPE, path: PATH_TYPE) -> nir.NIRGraph:
    """Load a NIR from a HDF/conn5 file."""
    with h5py.File(net_config, "r") as f:
        nir_graph = read_node(f["layer"])
    #nir_graph.input_type['input'] = nir_graph.input_type.pop('input_0')
    nir_graph = normalize_nir_graph(nir_graph, to_bytes_in_edges=True)
    #nir_graph.check_types()
    nir_graph.edges[3]=(b'input',0)
    nir_graph.edges[4]=(3,b'output')
    nir.write(path, nir_graph)


class Network:
    def __init__(self, path: typing.Union[str, pathlib.Path]) -> None:
        nir_graph = nir.read(path)
        self.graph = nir_graph
        # TODO: implement the NIR -> Lava conversion
        pass

def normalize_nir_graph(obj, to_bytes_in_edges=False, _in_edges=False):
    """
    Recursively normalize all 'input_#'/'output_#' to 'input'/'output'.
    Only convert 'input'/'output' to bytes inside the 'edges' field if to_bytes_in_edges is True.
    """
    def norm(x, in_edges):
        # Normalize input/output with optional _#
        if isinstance(x, str):
            if re.match(r'^input(_\d+)?$', x):
                return b'input' if (to_bytes_in_edges and in_edges) else 'input'
            if re.match(r'^output(_\d+)?$', x):
                return b'output' if (to_bytes_in_edges and in_edges) else 'output'
            return x
        if isinstance(x, bytes):
            try:
                x_str = x.decode()
            except Exception:
                return x
            if re.match(r'^input(_\d+)?$', x_str):
                return b'input' if (to_bytes_in_edges and in_edges) else 'input'
            if re.match(r'^output(_\d+)?$', x_str):
                return b'output' if (to_bytes_in_edges and in_edges) else 'output'
            return x
        return x

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == 'edges' and isinstance(v, list):
                # Recursively process everything inside edges
                new_dict[k] = normalize_nir_graph(v, to_bytes_in_edges, _in_edges=True)
            else:
                new_dict[norm(k, _in_edges)] = normalize_nir_graph(v, to_bytes_in_edges, _in_edges)
        return new_dict
    if isinstance(obj, list):
        return [normalize_nir_graph(v, to_bytes_in_edges, _in_edges) for v in obj]
    if isinstance(obj, tuple):
        return tuple(normalize_nir_graph(v, to_bytes_in_edges, _in_edges) for v in obj)
    if isinstance(obj, set):
        return {normalize_nir_graph(v, to_bytes_in_edges, _in_edges) for v in obj}
    if isinstance(obj, np.ndarray) and obj.dtype.kind == 'U':
        return obj
    if hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            setattr(obj, k, normalize_nir_graph(v, to_bytes_in_edges, _in_edges))
        return obj
    return norm(obj, _in_edges)