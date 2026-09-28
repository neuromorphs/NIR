"""
NIR primitives as equations.

Normative by reference: formalises the table in nir/docs/source/primitives.md.

Imports nothing from this package, and nothing from `nir` — the adapter in ingest.py owns that dependency so this table can be vendored upstream without a cycle.
"""

from __future__ import annotations

import nir
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import sympy as sp

t = sp.Symbol("t", positive=True)

v, i, x, y, z = sp.symbols("v i x y z")
tau, r, v_leak, v_threshold, w, delay = sp.symbols("tau r v_leak v_threshold w delay")


class SympySemantic:
    pass


@dataclass(frozen=True)
class Derivative(SympySemantic):
    """d(state)/dt = rhs"""

    state: sp.Symbol
    rhs: sp.Expr


@dataclass(frozen=True)
class Jump(SympySemantic):
    """lhs = rhs. No state, no time."""

    lhs: sp.Symbol
    rhs: sp.Expr


@dataclass(frozen=True)
class Delay(SympySemantic):
    """out(t) = src(t - by), `by` real-valued."""

    out: sp.Symbol
    src: sp.Symbol
    by: sp.Expr


class SemanticNode:
    node: nir.NIRNode


@dataclass(frozen=True)
class PrimitiveNode(SemanticNode):
    """"""

    eqs: Tuple[object, ...]
    output: sp.Symbol
    guard: Optional[sp.Expr] = None
    reset: Tuple[Jump, ...] = ()


@dataclass(frozen=True)
class SemanticGraph(SemanticNode):
    nodes: Dict[str, PrimitiveNode]
    edges: Tuple[Tuple[str, str], ...] = ()


def from_nir(node: nir.NIRNode, expr: sp.Expr) -> SemanticNode:
    if isinstance(node, nir.NIRGraph):
        converted = {name: from_nir(n, expr) for name, n in node.nodes.items()}
        return SemanticGraph(node=node, nodes=converted, edges=node.edges)
    elif node.__class__.__name__ in SEMANTICS:
        return PrimitiveNode(node, SEMANTICS[node.__class__.__name__])
    elif not isinstance(node, nir.NIRNode):
        raise ValueError("The node input is not a NIRNode")
    else:
        raise ValueError(
            f"No Sympy semantics found for node type {node.__class__.__name__}"
        )


# The semantics map
SEMANTICS: Dict[str, Callable[..., sp.Expr]] = {
    "LI": Derivative(v, ((v_leak - v) + r * i) / tau),
    "Threshold": Jump(z, sp.Heaviside(v - v_threshold)),
}


def Linear(*, w_=1.0) -> Primitive:
    """Stateless. Every rule is the identity on it — the pass-through case."""
    return Primitive(params=dict(w=w_), eqs=(Instant(y, w * x),), output=y)


def LIF(*, tau_=10e-3, r_=1.0, v_leak_=0.0, v_threshold_=1.0) -> Primitive:
    return Primitive(
        params=dict(tau=tau_, r=r_, v_leak=v_leak_, v_threshold=v_threshold_),
        eqs=(Deriv(v, ((v_leak - v) + r * i) / tau),),
        output=v,
        guard=v - v_threshold,
        reset=(Instant(v, v - v_threshold),),
    )


def Delay(*, delay_=3e-3) -> Primitive:
    """Not an ODE. Included precisely because it breaks the ODE-rule story.

    primitives.md names the parameter `tau`, colliding with LI's time constant;
    nir.ir.Delay names the field `delay`. Following the implementation.
    """
    return Primitive(params=dict(delay=delay_), eqs=(Delayed(y, x, delay),), output=y)


def Threshold(*, threshold_=1.0) -> Primitive:
    raise NotImplementedError(Threshold.__doc__)
