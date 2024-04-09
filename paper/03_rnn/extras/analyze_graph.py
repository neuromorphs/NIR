"""Analyze weight distribution of Braille graph."""

import matplotlib.pyplot as plt
import numpy as np

import nir

nir_model = nir.read("braille.nir")


def analyze_weights(weight, name):
    print("Analyze weight matrix", name)
    print(30 * "=")
    print("shape:", weight.shape)
    print("min:", np.min(weight))
    print("max:", np.max(weight))
    print("mean:", np.mean(weight))
    print("")
    plt.hist(weight, 21)
    plt.xlabel("weight")
    plt.ylabel("frequency")
    plt.title(name)
    plt.show()


for name, node in nir_model.nodes.items():
    if hasattr(node, "weight"):
        analyze_weights(node.weight, name)
