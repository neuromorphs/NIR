import matplotlib.pyplot as plt
import nir_reference_impl
import norse
import numpy as np
import snntorch as snn
import torch

# NOTE: this requires snntorch Pull-Request #246 and nirtorch/feature-state
from snntorch import export_nirtorch
from torch import nn


def compare_spike_counts(results):
    n_spikes_st = np.count_nonzero(results["spikes_snntorch"])
    n_spikes_ref = np.count_nonzero(results["spikes_ref"])
    n_spikes_norse = np.count_nonzero(results["spikes_norse"])
    print("Spikes snnTorch:", n_spikes_st)
    print("Spikes ref:     ", n_spikes_ref)
    print("Spikes Norse:   ", n_spikes_norse)


def plot_voltages(results, threshold=None):
    v_st = results["voltages_snntorch"]
    v_ref = results["voltages_ref"]
    v_norse = results["voltages_norse"]
    plt.plot(v_st, label="snntorch")
    plt.plot(v_ref, label="ref")
    plt.plot(v_norse, label="norse")
    if threshold:
        plt.axhline(y=threshold, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("time step")
    plt.ylabel("voltage")
    plt.legend()
    plt.show()


def compare_synaptic_to_nir(
    nrn_params: {},
    input_spikes: [],
    weight: float = 1.0,
    bias: float = 0.0,
    n_steps=0,
    do_plot=False,
):
    """Compare snntorch.Synaptic to NIR reference.

    Runs a single snntorch.Synaptic neuron with predefined stimulus and
    compares it to reference model.
    1. assembles input data from bias, input_spikes and weight
    2. create and run Synaptic model
    3. convert Synaptic model to NIR
    4. run the CubaLIF node of the NIR model with its reference implementation
       and same input as snntorch.

    Args:
        nrn_params(dict): snntorch.Synaptic neuron parameters. Supported keys:
            "alpha", "beta", "threshold", and "reset_mechanism".
        input_spikes(list): list of spike times as multiples of the timestep dt.
        weight(float): weight for input spikes
        bias(float): bias added in each timestep to input of Synaptic model
        n_steps(float): number of steps to run the model
        do_plot(bool): plot voltages

    Returns:
        dictionary with spikes and voltages of reference and snntorch
        keys: ["spikes_ref", "spikes_snntorch", "voltages_ref", "voltages_snntorch"]
    """
    #######################################
    # assemble input from spikes and bias #
    #######################################
    spk_in = torch.ones((n_steps, 1)) * bias

    for i in input_spikes:
        spk_in[i, 0] = weight

    ###############################################
    # snnTorch network with single Synaptic layer #
    ###############################################
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cuba = snn.Synaptic(
                alpha=[
                    nrn_params["alpha"]
                ],  # turn at least one parameter into list to set shape
                beta=nrn_params["beta"],
                threshold=nrn_params["threshold"],
                reset_mechanism=nrn_params["reset_mechanism"],
                reset_delay=False,  # enforce same reset behavior as in NIR
            )

        def forward(self, x):
            syn, mem = self.cuba.init_synaptic()
            spk_out = torch.zeros(1)
            syn_rec = []
            mem_rec = []
            spk_rec = []

            # Simulate neurons
            for step in range(n_steps):
                spk_out, syn, mem = self.cuba(x[step], syn, mem)
                spk_rec.append(spk_out)
                syn_rec.append(syn)
                mem_rec.append(mem)

            # convert lists to tensors
            spk_rec = torch.stack(spk_rec)
            syn_rec = torch.stack(syn_rec)
            mem_rec = torch.stack(mem_rec)

            return spk_rec, syn_rec, mem_rec

    # run snntorch model
    net = Net()
    spikes_st, i_st, v_st = net(spk_in)

    # export to nir
    nir_graph = export_nirtorch.to_nir(net, spk_in, ignore_dims=[0])
    nir_node = nir_graph.nodes["cuba"]

    ###################
    # Reference model #
    ###################
    dt = 0.0001  # this is used in snntorch.export_nirtorch.py (magic number)
    input_data = spk_in.detach().numpy()

    # run reference model
    ref_model = nir_reference_impl.CubaLIFImplementation(dt, nir_node)
    ref_results = nir_reference_impl.run_cuba_reference_model(ref_model, input_data)
    ref_spikes = ref_results["spikes"]
    ref_voltages = ref_results["voltages"]

    #########
    # Norse #
    #########

    norse_model = norse.torch.from_nir(nir_graph, dt=1e-4)

    def evaluate(x):
        out = []
        state = None
        v = []
        i = []
        for t in x:
            z, state = norse_model(t, state)
            out.append(z)
            cuba_state = state.state["cuba"][0][1]
            cuba_state = state.state["cuba"][0]
            v.append(cuba_state[2].v)
            i.append(cuba_state[1].v)
        return out, v, i

    out, v, i = evaluate(torch.Tensor(input_data))
    spikes_norse = torch.stack(out)
    voltages_norse = torch.stack(v)
    # currents_norse = torch.stack(i)

    results = {
        "spikes_ref": ref_spikes[:, 0],
        "spikes_snntorch": spikes_st.detach().numpy(),
        "spikes_norse": spikes_norse.detach().numpy(),
        "voltages_ref": ref_voltages[:, 0],
        "voltages_snntorch": v_st.detach().numpy(),
        "voltages_norse": voltages_norse.detach().numpy(),
    }

    if do_plot:
        plot_voltages(results, threshold=nrn_params["threshold"])

    return results


def test_convert_synaptic_negative_weight(do_plot=False):
    params = {"alpha": 0.7, "beta": 0.6, "threshold": 3, "reset_mechanism": "zero"}
    input_spikes = [10, 30, 40, 60, 70, 80, 90, 100, 120, 130, 150, 160, 190]
    compare_synaptic_to_nir(
        params, input_spikes, weight=-1, n_steps=200, do_plot=do_plot
    )


def test_convert_synaptic_sub_threshold(do_plot=False):
    params = {"alpha": 0.5, "beta": 0.7, "threshold": 6, "reset_mechanism": "subtract"}
    input_spikes = [40, 60, 70, 80, 100, 110, 120, 130, 140, 150, 160, 170]
    compare_synaptic_to_nir(
        params, input_spikes, weight=1, n_steps=200, do_plot=do_plot
    )


def test_convert_synaptic_reset_by_subtraction(do_plot=False):
    params = {"alpha": 0.9, "beta": 0.8, "threshold": 5, "reset_mechanism": "subtract"}
    input_spikes = np.arange(0, 200, 10, dtype=int)
    compare_synaptic_to_nir(
        params, input_spikes, weight=1, n_steps=200, do_plot=do_plot
    )


def test_convert_synaptic_reset_to_zero(do_plot=False):
    params = {"alpha": 0.8, "beta": 0.9, "threshold": 5, "reset_mechanism": "zero"}
    input_spikes = np.arange(0, 200, 10, dtype=int)
    compare_synaptic_to_nir(
        params, input_spikes, weight=1, n_steps=200, do_plot=do_plot
    )


def test_convert_synaptic_reset_by_subtraction_const_I(do_plot=False):
    params = {
        "alpha": 0.9,
        "beta": 0.8,
        "threshold": 5.0,
        "reset_mechanism": "subtract",
    }
    input_spikes = np.arange(200, dtype=int)
    weight = 0.11
    results = compare_synaptic_to_nir(
        params, input_spikes, weight, n_steps=200, do_plot=do_plot
    )
    compare_spike_counts(results)


def compare_transfer_function(do_plot=False):
    params = {
        "alpha": 0.9,
        "beta": 0.8,
        "threshold": 5.0,
        "reset_mechanism": "subtract",
    }
    input_spikes = []
    biases = np.arange(0.0, 1.0, 0.02)
    spikes_st = np.zeros_like(biases)
    spikes_ref = np.zeros_like(biases)
    spikes_norse = np.zeros_like(biases)
    n_steps = 200
    for i, bias in enumerate(biases):
        results = compare_synaptic_to_nir(
            params, input_spikes, bias=bias, n_steps=n_steps, do_plot=False
        )
        compare_spike_counts(results)
        spikes_st[i] = np.count_nonzero(results["spikes_snntorch"])
        spikes_ref[i] = np.count_nonzero(results["spikes_ref"])
        spikes_norse[i] = np.count_nonzero(results["spikes_norse"])

    if do_plot:
        plt.plot(biases, spikes_st / n_steps, label="snntorch")
        plt.plot(biases, spikes_ref / n_steps, label="ref")
        plt.plot(biases, spikes_norse / n_steps, label="norse")
        plt.xlabel("bias")
        plt.ylabel("avg. spikes per timestep")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # test_convert_synaptic_sub_threshold(do_plot=True)
    # test_convert_synaptic_negative_weight(do_plot=True)
    # test_convert_synaptic_reset_by_subtraction(do_plot=True)
    # test_convert_synaptic_reset_to_zero(do_plot=True)
    test_convert_synaptic_reset_by_subtraction_const_I(do_plot=True)
    compare_transfer_function(True)
