"""
NIR primitives as equations, formalising docs/source/primitives.md.

sympy is NOT a dependency of nir: it is declared in the `sem` dependency group,
this module is its only importer, and nir/ir/__init__.py MUST NOT import it.

    from nir.ir import semantics
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

import sympy as sp

from .graph import NIRGraph
from .node import NIRNode

t = sp.Symbol("t")  # no assumptions: positive=True zeroes DiracDelta(t)
v, i, x, y, z = sp.symbols("v i x y z")
tau, r, delay = sp.symbols("tau r delay", positive=True)
v_leak, v_threshold, v_reset, w, b = sp.symbols("v_leak v_threshold v_reset w b")


class Semantic:
    """A single equation."""

    @property
    def defines(self) -> sp.Symbol:
        """The symbol this equation determines."""
        raise NotImplementedError


@dataclass(frozen=True)
class Derivative(Semantic):
    """d(state)/dt = rhs."""

    state: sp.Symbol
    rhs: sp.Expr

    @property
    def defines(self) -> sp.Symbol:
        return self.state


@dataclass(frozen=True)
class Algebraic(Semantic):
    """lhs = rhs."""

    lhs: sp.Symbol
    rhs: sp.Expr

    @property
    def defines(self) -> sp.Symbol:
        return self.lhs


@dataclass(frozen=True)
class Delay(Semantic):
    """output(t) = input(t - shift), `shift` real-valued."""

    output: sp.Symbol
    input: sp.Symbol
    shift: sp.Expr

    @property
    def defines(self) -> sp.Symbol:
        return self.output


@dataclass(frozen=True)
class Crossing(Semantic):
    """A unit atom at each upward sign change of `arg`."""

    out: sp.Symbol
    arg: sp.Expr

    @property
    def defines(self) -> sp.Symbol:
        return self.out


@dataclass(frozen=True)
class Primitive:
    """One NIR Primitive in the form of equations and bound symbols."""

    equations: Tuple[Semantic, ...]
    # Symbol -> nir field name, listing only the symbols whose names disagree
    # with the field they bind to. An unlisted symbol binds to its own name, so
    # `r` needs no entry while Linear's `w` does
    # A symbol that binds to no field at all is an INPUT
    aliases: Mapping[sp.Symbol, str] = MappingProxyType({})

    def __post_init__(self):
        # frozen protects the field, not a dict's contents, and from_nir hands
        # this same object to every node of the type.
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))

    def field_of(self, symbol: sp.Symbol) -> str:
        """The nir field `symbol` binds to. Its own name unless aliased."""
        return self.aliases.get(symbol, symbol.name)


@dataclass(frozen=True)
class SemanticNode:
    node: NIRNode


@dataclass(frozen=True)
class SemanticLeaf(SemanticNode):
    """One NIR node and the equations it means."""

    semantics: Primitive


@dataclass(frozen=True)
class SemanticGraph(SemanticNode):
    nodes: Dict[str, SemanticNode] = field(default_factory=dict)
    edges: Tuple[Tuple[str, str], ...] = ()
    boundary: Tuple[str, ...] = ()


_BOUNDARY = ("Input", "Output")


def from_nir(node: NIRNode) -> SemanticNode:
    """Attach equations to a NIR graph or node, recursing into NIRGraph."""
    if isinstance(node, NIRGraph):
        inner, skipped = {}, []
        for name, n in node.nodes.items():
            if type(n).__name__ in _BOUNDARY:
                skipped.append(name)
            else:
                inner[name] = from_nir(n)
        return SemanticGraph(
            node=node,
            nodes=inner,
            edges=tuple(tuple(e) for e in node.edges),
            boundary=tuple(skipped),
        )
    if not isinstance(node, NIRNode):
        raise ValueError(f"not a NIRNode: {type(node).__name__}")
    kind = type(node).__name__
    if kind not in SEMANTICS:
        raise ValueError(
            f"no sympy semantics for {kind} (have: {', '.join(sorted(SEMANTICS))})"
        )
    return SemanticLeaf(node=node, semantics=SEMANTICS[kind])


_SCOPE_SEP = "."
_GLOBAL = frozenset({t})


def qualify(symbol: sp.Symbol, scope: str) -> sp.Symbol:
    """`v` in scope "n1" becomes `n1.v`, assumptions preserved."""
    if not scope:
        raise ValueError("scope must be a non-empty node name")
    return sp.Symbol(f"{scope}{_SCOPE_SEP}{symbol.name}", **symbol.assumptions0)


def rename(semantic: Semantic, scope: str) -> Semantic:
    """Every symbol of `semantic`, bound or free, moved into `scope`."""
    subs: Dict[sp.Symbol, sp.Symbol] = {}
    for f in fields(semantic):
        value = getattr(semantic, f.name)
        if isinstance(value, sp.Expr):
            for s in value.free_symbols - _GLOBAL:
                subs.setdefault(s, qualify(s, scope))
    renamed = {
        f.name: getattr(semantic, f.name).xreplace(subs)
        for f in fields(semantic)
        if isinstance(getattr(semantic, f.name), sp.Expr)
    }
    return replace(semantic, **renamed)


def _free(semantic: Semantic) -> set:
    """Every symbol mentioned by `semantic`, time excluded."""
    return set().union(
        *(
            getattr(semantic, f.name).free_symbols
            for f in fields(semantic)
            if isinstance(getattr(semantic, f.name), sp.Expr)
        )
    ) - _GLOBAL


def ports(node: SemanticLeaf) -> Tuple[Tuple[sp.Symbol, ...], sp.Symbol]:
    """(inputs, output) of `node`.

    Every symbol an equation mentions is one of three things:

      * DEFINED -- it is what some equation of this node determines. That covers
        a state, an output, and a symbol bound by a sibling equation.
      * a PARAMETER -- it binds to a field of the nir node, by name or by alias.
      * an INPUT -- whatever is left.
    """
    sem = node.semantics
    params = {f.name for f in fields(type(node.node))}
    defined = {e.defines for e in sem.equations}
    free = set().union(*(_free(e) for e in sem.equations))
    inputs = tuple(sorted(free - defined - {s for s in free if sem.field_of(s) in params}, key=str))

    fires = [e.defines for e in sem.equations if isinstance(e, Crossing)]
    if len(fires) > 1:
        raise ValueError(f"{type(node.node).__name__}: {len(fires)} Crossings, no single output")
    if fires:
        return inputs, fires[0]
    if len(sem.equations) != 1:
        raise ValueError(f"{type(node.node).__name__}: {len(sem.equations)} equations, no Crossing")
    return inputs, sem.equations[0].defines


def system(graph: SemanticGraph) -> Tuple[Semantic, ...]:
    """`graph` as one system of equations, each node renamed into its own scope."""
    raise NotImplementedError(system.__doc__)


SEMANTICS: Dict[str, Primitive] = {
    "I":         Primitive((Derivative(v, r * i),)),
    "LI":        Primitive((Derivative(v, ((v_leak - v) + r * i) / tau),)),
    "Linear":    Primitive((Algebraic(y, w * x),), {w: "weight"}),
    "Scale":     Primitive((Algebraic(y, w * x),), {w: "scale"}),
    "Delay":     Primitive((Delay(y, x, delay),)),
    "Threshold": Primitive(
        (Crossing(z, v - v_threshold),), {v_threshold: "threshold"}
    ),
    "IF": Primitive(
        (
            Derivative(v, r * i + (v_reset - v_threshold) * z),
            Crossing(z, v - v_threshold),
        )
    ),
    "LIF": Primitive(
        (
            Derivative(v, ((v_leak - v) + r * i) / tau + (v_reset - v_threshold) * z),
            Crossing(z, v - v_threshold)
        )
    )
}
