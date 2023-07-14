import nengo
import numpy as np
import nir

model = nengo.Network(seed=3)
with model:
    model.config[nengo.Ensemble].neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0)

    x = nengo.Ensemble(n_neurons=600, dimensions=3, radius=30)

    synapse = 0.1

    def lorenz(x):
        sigma = 10
        beta = 8.0 / 3
        rho = 28

        dx0 = -sigma * x[0] + sigma * x[1]
        dx1 = -x[0] * x[2] - x[1]
        dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho

        return [dx0 * synapse + x[0], dx1 * synapse + x[1], dx2 * synapse + x[2]]

    nengo.Connection(x, x, synapse=synapse, function=lorenz)



























def nengo_to_nir(model):    
    model2 = nengo.Network()
    model2.networks.append(model)
    for p in model.all_probes:
        if isinstance(p.target, nengo.Ensemble):
            with model2:
                output = nengo.Node(lambda t, x:x, size_in=p.target.dimensions)
                nengo.Connection(p.target, output, synapse=p.synapse)
        elif isinstance(p.target, nengo.Node) and p.target.output is None:
            with model2:
                output = nengo.Node(lambda t, x:x, size_in=p.target.size_out)
                nengo.Connection(p.target, output, synapse=p.synapse)
        else:
            raise Exception(f'Unhandled Probe {p}')

    sim = nengo.simulator.Simulator(model2)
    nengo2nir = {}
    nir_nodes = []
    nir_edges = []
    
    for node in model2.all_nodes:
        if node.size_in == 0:
            n = nir.Input(shape=[node.size_out])
            nengo2nir[node] = (None, n)
            nir_nodes.append(n)
        elif node.output is None:
            N = node.size_in
            n = nir.Affine(weight=np.eye(N), bias=np.tile(0, N))
            nengo2nir[node] =(n, n)
            nir_nodes.append(n)
        else:
            n = nir.Output()
            nengo2nir[node] = (n, None)
            nir_nodes.append(n)
    for ens in model2.all_ensembles:
        assert isinstance(ens.neuron_type, nengo.LIF) 
        assert ens.neuron_type.tau_ref == 0
        enc = sim.data[ens].scaled_encoders
        bias = sim.data[ens].bias
        
        e = nir.Affine(weight=enc, bias=bias)
        nir_nodes.append(e)
        N = ens.n_neurons
        lif = nir.LIF(tau=np.tile(ens.neuron_type.tau_rc, N), 
                      v_threshold=np.tile(1, N), 
                      r=np.tile(1, N), 
                      v_leak=np.tile(0, N))
        nir_nodes.append(lif)
        nir_edges.append((len(nir_nodes)-2, len(nir_nodes)-1))
        nengo2nir[ens] = e, lif
    for conn in model2.all_connections:
        w = sim.data[conn].weights
    
        for i in range(len(nir_nodes)):
            if nir_nodes[i] is nengo2nir[conn.pre_obj][1]:
                source_index = i
                break
        for i in range(len(nir_nodes)):
            if nir_nodes[i] is nengo2nir[conn.post_obj][0]:
                target_index = i
                break    
        #source_index = nir_nodes.index(nengo2nir[conn.pre_obj][1])
        #target_index = nir_nodes.index(nengo2nir[conn.post_obj][0])
        
        if conn.pre_slice != slice(None, None, None):
            t = np.eye(conn.pre_obj.size_out)[conn.pre_slice,:]
            lin = nir.Affine(weight=t, bias=np.tile(0, t.shape[0]))
            nir_nodes.append(lin)
            nir_edges.append((source_index, len(nir_nodes)-1))
            source_index = len(nir_nodes)-1
        
        if w is not None:
            if w.shape == ():
                w = np.eye(conn.size_in)*w
            lin = nir.Affine(weight=w, bias=np.tile(0, w.shape[0]))
            nir_nodes.append(lin)
            nir_edges.append((source_index, len(nir_nodes)-1))
            source_index = len(nir_nodes)-1
        #if not isinstance(conn.transform, nengo.transforms.NoTransform):
        #    t = sim.data[conn].transform.sample()
        #    if t.shape == ():
        #        t.shape = (1,1)
        #    if t.shape == (1,1) and conn.size_in > 1:
        #        t = np.eye(conn.size_in)*t[0,0]
        #    print(t.shape)
        #    lin = nir.(weight=t, bias=np.tile(0, t.shape[1]))
        #    nir_nodes.append(lin)
        #    nir_edges.append((source_index, len(nir_nodes)-1))
        #    source_index = len(nir_nodes)-1
        if conn.synapse is not None:
            assert isinstance(conn.synapse, nengo.synapses.Lowpass)
            N = conn.size_out
            ir = nir.LI(tau=np.tile(conn.synapse.tau, N), 
                        r=np.tile(1, N), 
                        v_leak=np.tile(0, N))
            nir_nodes.append(ir)
            nir_edges.append((source_index, len(nir_nodes)-1))
            source_index = len(nir_nodes)-1
            
        if conn.post_slice != slice(None, None, None):
            t = np.eye(conn.post_obj.size_in)[:,conn.post_slice]
            lin = nir.Affine(weight=t, bias=np.tile(0, t.shape[0]))
            nir_nodes.append(lin)
            nir_edges.append((source_index, len(nir_nodes)-1))
            source_index = len(nir_nodes)-1
            
            
        nir_edges.append((source_index, target_index))
    return nir.NIRGraph(nodes=nir_nodes, edges=nir_edges)           
        
def nir_to_nengo(n):
    nengo_map = []
    
    model = nengo.Network()
    with model:
        filters = {}
        for i, obj in enumerate(n.nodes):
            if isinstance(obj, nir.Input):
                node = nengo.Node(np.zeros(obj.shape), label=f'Input {i} {obj.shape}')
                nengo_map.append(node)
            elif isinstance(obj, nir.LIF):
                N = obj.tau.flatten().shape[0]
                ens = nengo.Ensemble(n_neurons=N, dimensions=1, label=f'LIF {i}',
                                     neuron_type=nengo.RegularSpiking(nengo.LIFRate(tau_rc=obj.tau[0], tau_ref=0)),
                                     #neuron_type=nengo.LIF(tau_rc=obj.tau[0], tau_ref=0),
                                     gain=np.ones(N), bias=np.zeros(N))
                nengo_map.append(ens.neurons)
            elif isinstance(obj, nir.LI):
                filt = nengo.Node(lambda t,x:x, size_in=obj.tau.flatten().shape[0], label=f'LI {i} {obj.tau.shape}')
                filters[filt] = nengo.synapses.Lowpass(obj.tau[0])
                nengo_map.append(filt)
            elif isinstance(obj, nir.Affine):
                w = nengo.Node(lambda t, x, obj=obj: obj.weight @ x + obj.bias, size_in=obj.weight.shape[1], size_out=obj.weight.shape[0], label=f'({obj.weight.shape[0]}x{obj.weight.shape[1]})')
                nengo_map.append(w)
            elif isinstance(obj, nir.Output):
                nengo_map.append(None)  # because the NIR spec doesn't tell me the size, so I can't create this yet
            else:
                raise Exception(f'Unknown NIR object: {obj}')
        for (pre, post) in n.edges:
            if nengo_map[post] is None:
                output = nengo.Node(lambda t, x:x, size_in=nengo_map[pre].size_out, label=f'Output {post}')
                nengo_map[post] = output
            synapse = filters.get(nengo_map[post], None)
            
            if nengo_map[pre].size_out != nengo_map[post].size_in:
                print('Error')
                print('pre', nengo_map[pre])
                print('post', nengo_map[post])
                1/0
                
            else:
                nengo.Connection(nengo_map[pre], nengo_map[post], synapse=synapse)
                
    return model

n = nengo_to_nir(model)
model = nir_to_nengo(n)

