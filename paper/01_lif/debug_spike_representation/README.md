# Example for debugging the spike representation in NIR

## Problem
Currently, the NIR neurons output a 1 when the neuron spikes.
In the discrete-time backends (i.e. all we have so far) however the strength of
the spike (its effect on the postsynaptic membrane potential depends on the
discretization timestep `dt`: The larger `dt`, the larger the effect.
The reason is that we treat everything (also incoming spikes), as scalar
signals which are constant over the interval `dt`. 

## Example showing the problem

In file `create_model.py` we create a 2 LIF neuron network.
    LIF1 -> Linear -> LIF2
    
On purpose, there is no external input. Instead, LIF1 is made self-spiking by
setting the `v_leak` above threshold.
Spikes are sent to the `LIF2` and increase the membrane potential. LIF2 does
never reach the threshold, so what matters are the subthreshold dynamics.

The notebook `Run on SpiNNaker2.ipynb` runs this model for different `dt`
values with the SpiNNaker2-brian2 backend.

While the dynamics of the LIF1 neuron is the same in all cases (which shows
that the parameter translation works for different `dt`), the dynamics differ
for `LIF2`:
The smaller `dt`, the smaller is the effect of the post-synaptic potential!
Explanation: see above

**TODO: it would be great if someone could replicate this in another backend**

## Solution:
In NIR, we should consider spikes as Dirac pulses `\delta(t)` whose integral is
1.
by enforcing that *The integral over the spike needs to be 1*, the conversion
to the frameworks might need to be changed (e.g. by scaling weights, sending
other values than 1s etc..)
