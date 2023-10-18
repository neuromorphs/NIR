import lava.lib.dl.slayer as slayer
import numpy as np
import nir
import torch


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
        elif self.end_dim == len(x.shape)-1:
            end_dim = -2
        # if end_dim != self.end_dim:
        #     print(f'FLATTEN: changed end_dim from {self.start_dim} to {end_dim}')
        return x.flatten(self.start_dim, end_dim)


def convert_node_to_lava_dl_element(node, scale_v_thr=1.0):
    debug_conv = False
    debug_pool = False
    debug_if = False
    debug_affine = False

    if isinstance(node, nir.ir.Conv2d):
        assert np.abs(node.bias).sum() == 0.0, 'bias not supported in lava-dl'
        out_features = node.weight.shape[0]
        in_features = node.weight.shape[1]
        kernel_size = (node.weight.shape[2], node.weight.shape[3])
        # stride = int(node.stride[0])
        # assert node.stride[0] == node.stride[1], 'stride must be the same in both dimensions'
        if debug_conv:
            print(f'Conv2d with weights of shape {node.weight.shape}:')
            print(f'\t{in_features} in, {out_features} out, kernel {kernel_size}')
            print(f'\tstride {node.stride}, padding {node.padding}, dilation {node.dilation}')
            print(f'\tgroups {node.groups}')
        conv_synapse_params = dict(
            in_features=in_features, out_features=out_features,
            kernel_size=kernel_size, stride=node.stride, padding=node.padding,
            dilation=node.dilation, groups=node.groups,
            weight_scale=1, weight_norm=False, pre_hook_fx=None
        )
        conv = slayer.synapse.Conv(**conv_synapse_params)
        conv.weight.data = torch.from_numpy(node.weight.reshape(conv.weight.shape))
        return conv

    elif isinstance(node, nir.ir.SumPool2d):
        if debug_pool:
            print(f'SumPool2d: kernel {node.kernel_size} pad {node.padding}, stride {node.stride}')
        pool_synapse_params = dict(
            kernel_size=node.kernel_size, stride=node.stride, padding=node.padding, dilation=1,
            weight_scale=1, weight_norm=False, pre_hook_fx=None
        )
        return slayer.synapse.Pool(**pool_synapse_params)

    elif isinstance(node, nir.ir.IF):
        assert len(np.unique(node.v_threshold)) == 1, 'v_threshold must be the same for all neurons'
        assert len(np.unique(node.r)) == 1, 'resistance must be the same for all neurons'
        v_thr = np.unique(node.v_threshold)[0]
        resistance = np.unique(node.r)[0]
        v_thr_eff = v_thr * resistance * scale_v_thr
        if debug_if:
            print(f'IF with v_thr={v_thr}, R={resistance} -> eff. v_thr={v_thr_eff}')
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
        assert np.abs(node.bias).sum() == 0.0, 'bias not supported in lava-dl'
        weight = node.weight
        out_neurons = weight.shape[0]
        in_neurons = weight.shape[1]
        if debug_affine:
            print(f'Affine: weight shape: {weight.shape}')
        dense = slayer.synapse.Dense(
            in_neurons=in_neurons, out_neurons=out_neurons,
            weight_scale=1, weight_norm=False, pre_hook_fx=None
        )
        dense.weight.data = torch.from_numpy(weight.reshape(dense.weight.shape))
        return dense

    elif isinstance(node, nir.ir.Flatten):
        return Flatten(node.start_dim, node.end_dim)

    else:
        print("UNSUPPORTED")


def get_next_node_key(node_key, edges):
    possible_next_node_keys = [edge[1] for edge in edges if edge[0] == node_key]
    assert len(possible_next_node_keys) <= 1, 'branching networks are not supported'
    if len(possible_next_node_keys) == 0:
        return None
    else:
        return possible_next_node_keys[0]


class NIR2LavaDLNetwork(torch.nn.Module):
    def __init__(self, module_list, jens_order=False):
        super(NIR2LavaDLNetwork, self).__init__()
        if jens_order:
            new_list = []
            assert len(module_list) == 13
            for i in [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 2, 3, 4]:
                new_list.append(module_list[i])
            self.blocks = torch.nn.ModuleList(new_list)
        else:
            self.blocks = torch.nn.ModuleList(module_list)

    def forward(self, spike):
        for block in self.blocks:
            if isinstance(block, torch.nn.Module):
                spike = block(spike)
            else:
                raise Exception('Unknown block type')
        return spike


def nir_to_lava_dl(graph, scale_v_thr=1.0, debug=False):
    node_key = 'input'
    visited_node_keys = [node_key]
    module_list = []
    while get_next_node_key(node_key, graph.edges) is not None:
        node_key = get_next_node_key(node_key, graph.edges)
        node = graph.nodes[node_key]
        assert node_key not in visited_node_keys, 'cycling NIR graphs are not supported'
        visited_node_keys.append(node_key)
        if debug:
            print(f'node {node_key}: {type(node).__name__}')
        if node_key == 'output':
            continue
        module_list.append(convert_node_to_lava_dl_element(node, scale_v_thr=scale_v_thr))

    assert len(visited_node_keys) == len(graph.nodes), 'not all nodes visited'

    return NIR2LavaDLNetwork(module_list)
