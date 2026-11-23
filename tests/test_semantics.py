"""Tests for nir.ir.semantics.

sympy is in the `sem` dependency group, not in nir's dependencies, so the whole
module skips when it is absent rather than failing the vanilla suite.
"""

import dataclasses

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

import nir  # noqa: E402
from nir.ir import __all_ir  # noqa: E402
from nir.ir import semantics as sem  # noqa: E402

ONE = np.array([1.0])


def mock_li(*shape):
    return nir.LI(
        tau=np.ones(shape),
        r=np.ones(shape),
        v_leak=np.zeros(shape),
    )


def mock_threshold(*shape):
    return nir.Threshold(threshold=np.ones(shape))


# --- the table is exhaustive -------------------------------------------------
# This is the test that makes the table normative: a primitive is either
# formalised or explicitly declared absent with a reason, and a newly added
# primitive fails here until someone says which it is.


def test_every_primitive_is_either_formalised_or_declared_absent():
    missing = [
        name
        for name in __all_ir
        if name not in sem.SEMANTICS and name not in sem.DECLARED_ABSENT
    ]
    assert missing == []


def test_nothing_is_both_formalised_and_declared_absent():
    assert set(sem.SEMANTICS) & set(sem.DECLARED_ABSENT) == set()


def test_declared_absent_names_real_primitives():
    unknown = [name for name in sem.DECLARED_ABSENT if name not in __all_ir]
    assert unknown == []


def test_every_declared_absence_carries_a_reason():
    assert all(reason.strip() for reason in sem.DECLARED_ABSENT.values())


def test_boundary_nodes_are_declared_absent():
    for name in sem._BOUNDARY:
        assert name in sem.DECLARED_ABSENT


# --- assumptions -------------------------------------------------------------


@pytest.mark.parametrize("symbol", [sem.tau, sem.r, sem.delay])
def test_genuinely_positive_parameters_are_positive(symbol):
    assert symbol.is_positive


@pytest.mark.parametrize("symbol", [sem.v_leak, sem.v_threshold, sem.w])
def test_signed_parameters_are_not_assumed_positive(symbol):
    """v_leak < 0 at mV scale, w < 0 when inhibitory, v_threshold signed with
    v_leak. is_positive is None (unknown), not False."""
    assert symbol.is_positive is None


@pytest.mark.parametrize("symbol", [sem.v, sem.i, sem.x, sem.y, sem.z])
def test_states_and_ports_carry_no_assumption(symbol):
    assert symbol.assumptions0 == {"commutative": True}


def test_time_is_not_positive_so_an_event_at_the_origin_survives():
    """positive=True on t makes DiracDelta(t) collapse to 0, deleting a spike at
    t = 0. Regression guard for the one assumption that must not be added."""
    assert sp.DiracDelta(sem.t) != 0
    assert sp.DiracDelta(sp.Symbol("t", positive=True)) == 0


def test_positive_potentials_would_spoil_the_emitted_shape():
    """The reason the split is by quantity and not by convenience: a positive
    BASE keeps the exponent inside the log. With tau positive and the potentials
    sign-free, the solved first-crossing time comes out in the readable form."""
    t = sem.t
    v0_signed, theta_signed = sp.symbols("v0 theta")
    v0_pos, theta_pos = sp.symbols("v0 theta", positive=True)

    signed = sp.solve(sp.Eq(v0_signed * sp.exp(-t / sem.tau), theta_signed), t)
    assert signed == [sem.tau * sp.log(v0_signed / theta_signed)]

    positive = sp.solve(sp.Eq(v0_pos * sp.exp(-t / sem.tau), theta_pos), t)
    assert positive == [sp.log((v0_pos / theta_pos) ** sem.tau)]
    assert positive != signed


# --- the equations themselves ------------------------------------------------


def test_li_matches_primitives_md():
    """tau * dv/dt = (v_leak - v) + R*I."""
    li = sem.SEMANTICS["LI"].equations[0]
    assert isinstance(li, sem.Derivative)
    assert li.state == sem.v
    residual = sem.tau * li.rhs - ((sem.v_leak - sem.v) + sem.r * sem.i)
    assert sp.simplify(residual) == 0


def test_integrator_matches_primitives_md():
    """dv/dt = R*I: the leak-free limit, and NOT merely LI with tau = 1."""
    integrator = sem.SEMANTICS["I"].equations[0]
    assert isinstance(integrator, sem.Derivative)
    assert integrator.state == sem.v
    assert sp.simplify(integrator.rhs - sem.r * sem.i) == 0
    assert sem.v not in integrator.rhs.free_symbols


def test_linear_and_scale_are_the_same_equation():
    """Different nir nodes, one equation. The table is anonymous, so they are
    the same object -- which is what `system` has to undo per node."""
    assert sem.SEMANTICS["Linear"].equations == sem.SEMANTICS["Scale"].equations
    linear = sem.SEMANTICS["Linear"].equations[0]
    assert isinstance(linear, sem.Algebraic)
    assert linear.lhs == sem.y
    assert sp.simplify(linear.rhs - sem.w * sem.x) == 0


def test_delay_is_the_only_form_with_a_shift():
    delayed = sem.SEMANTICS["Delay"].equations[0]
    assert isinstance(delayed, sem.Delay)
    assert (delayed.output, delayed.input, delayed.shift) == (sem.y, sem.x, sem.delay)
    assert delayed.shift.is_positive


def test_threshold_is_a_transition_not_a_level():
    """The resolved spec question: an edge, so no Heaviside anywhere in it."""
    threshold = sem.SEMANTICS["Threshold"].equations[0]
    assert isinstance(threshold, sem.Crossing)
    assert threshold.out == sem.z
    assert sp.simplify(threshold.arg - (sem.v - sem.v_threshold)) == 0
    assert not threshold.arg.has(sp.Heaviside)


# --- the Transition form -----------------------------------------------------


def test_as_distribution_is_delta_times_the_chain_rule_factor():
    threshold = sem.SEMANTICS["Threshold"].equations[0]
    flow = sp.Symbol("flow")
    expr = threshold.as_distribution({sem.v: flow})
    assert sp.simplify(expr - sp.DiracDelta(threshold.arg) * flow) == 0


def test_as_distribution_ignores_states_absent_from_arg():
    """d(arg)/dv_other = 0, so an unrelated state contributes nothing."""
    threshold = sem.SEMANTICS["Threshold"].equations[0]
    flow, other = sp.symbols("flow other")
    with_other = threshold.as_distribution({sem.v: flow, sem.i: other})
    without = threshold.as_distribution({sem.v: flow})
    assert sp.simplify(with_other - without) == 0


def test_transition_has_unit_amplitude_where_the_prose_form_does_not():
    """The Jacobian claim, as an integral over a concrete ramp.

    v(t) = a*t crossing theta once. delta(v - theta) integrates to 1/a -- a
    slower crossing emitting a LARGER spike -- while the Transition's d(arg)/dt
    factor cancels the 1/|g'| from the composition rule and integrates to 1,
    independent of a.
    """
    t = sem.t
    a, theta = sp.symbols("a theta", positive=True)
    ramp = {sem.v: a * t, sem.v_threshold: theta}

    prose = sp.DiracDelta(sem.v - sem.v_threshold).subs(ramp)
    assert sp.integrate(prose, (t, -sp.oo, sp.oo)) == 1 / a

    edge = sem.SEMANTICS["Threshold"].equations[0].as_distribution({sem.v: a}).subs(ramp)
    assert sp.integrate(edge, (t, -sp.oo, sp.oo)) == 1


def test_transition_sign_follows_the_crossing_direction():
    """A downward crossing emits -1: the direction is in the sign, not a case
    split, which is what keeps the form a derivative."""
    t = sem.t
    a, theta = sp.symbols("a theta", positive=True)
    falling = {sem.v: -a * t, sem.v_threshold: -theta}

    edge = sem.SEMANTICS["Threshold"].equations[0].as_distribution({sem.v: -a}).subs(falling)
    assert sp.integrate(edge, (t, -sp.oo, sp.oo)) == -1


# --- from_nir ----------------------------------------------------------------


def test_from_nir_primitive_shares_the_table_object():
    """Anonymity invariant: no copy, no rename. Two LI nodes alias `v`, and that
    is deliberate -- `system` is where it gets resolved."""
    first, second = sem.from_nir(mock_li(2)), sem.from_nir(mock_li(3))
    assert isinstance(first, sem.SemanticLeaf)
    assert first.semantics is sem.SEMANTICS["LI"]
    assert first.semantics is second.semantics


def test_from_nir_leaves_guard_and_reset_empty():
    """Nothing in the table sets them yet; they are candidate (A) for the merge."""
    node = sem.from_nir(mock_li(2))
    assert node.guard is None
    assert node.reset == ()


def test_from_nir_recurses_into_a_graph():
    graph = nir.NIRGraph(
        nodes={"li": mock_li(2), "th": mock_threshold(2)},
        edges=[("li", "th")],
    )
    result = sem.from_nir(graph)
    assert isinstance(result, sem.SemanticGraph)
    assert set(result.nodes) == {"li", "th"}
    assert isinstance(result.nodes["li"].semantics.equations[0], sem.Derivative)
    assert isinstance(result.nodes["th"].semantics.equations[0], sem.Crossing)
    assert result.node is graph


def test_from_nir_collects_boundary_nodes_without_equations():
    graph = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type=np.array([2])),
            "li": mock_li(2),
            "out": nir.Output(output_type=np.array([2])),
        },
        edges=[("in", "li"), ("li", "out")],
    )
    result = sem.from_nir(graph)
    assert set(result.nodes) == {"li"}
    assert set(result.boundary) == {"in", "out"}


def test_boundary_names_still_appear_in_edges():
    """Dropping the equation must not drop the connectivity: a consumer has to
    see that the edge went through a node with no semantics."""
    graph = nir.NIRGraph(
        nodes={"in": nir.Input(input_type=np.array([2])), "li": mock_li(2)},
        edges=[("in", "li")],
    )
    result = sem.from_nir(graph)
    assert ("in", "li") in result.edges
    assert "in" in result.boundary


def test_edges_are_hashable_tuples():
    graph = nir.NIRGraph(nodes={"li": mock_li(2)}, edges=[])
    assert isinstance(sem.from_nir(graph).edges, tuple)
    graph = nir.NIRGraph(nodes={"a": mock_li(2), "b": mock_li(2)}, edges=[("a", "b")])
    assert hash(sem.from_nir(graph).edges)


def test_from_nir_refuses_an_unsemantified_primitive():
    with pytest.raises(ValueError, match="no sympy semantics for Affine"):
        sem.from_nir(nir.Affine(weight=np.ones((2, 2)), bias=np.ones(2)))


def test_from_nir_refuses_a_non_node():
    with pytest.raises(ValueError, match="not a NIRNode"):
        sem.from_nir(object())


def test_from_nir_refuses_an_unsemantified_primitive_inside_a_graph():
    graph = nir.NIRGraph(
        nodes={"li": mock_li(2), "flat": nir.Flatten(np.array([2]), 0, 0)},
        edges=[("li", "flat")],
    )
    with pytest.raises(ValueError, match="no sympy semantics for Flatten"):
        sem.from_nir(graph)


# --- scoping -----------------------------------------------------------------


def test_qualify_preserves_assumptions():
    """A renamed tau is still positive; the shape argument above depends on it."""
    scoped = sem.qualify(sem.tau, "n1")
    assert scoped.name == "n1.tau"
    assert scoped.is_positive
    assert sem.qualify(sem.v_leak, "n1").is_positive is None


def test_qualify_rejects_an_empty_scope():
    with pytest.raises(ValueError, match="non-empty"):
        sem.qualify(sem.v, "")


def test_qualify_is_injective_across_scopes():
    assert sem.qualify(sem.v, "n1") != sem.qualify(sem.v, "n2")
    assert sem.qualify(sem.v, "n1") == sem.qualify(sem.v, "n1")


def test_rename_moves_every_symbol_of_a_derivative():
    renamed = sem.rename(sem.SEMANTICS["LI"].equations[0], "n1")
    assert isinstance(renamed, sem.Derivative)
    assert renamed.state == sem.qualify(sem.v, "n1")
    assert all(s.name.startswith("n1.") for s in renamed.rhs.free_symbols)
    assert renamed.rhs.free_symbols == {
        sem.qualify(s, "n1") for s in sem.SEMANTICS["LI"].equations[0].rhs.free_symbols
    }


def test_rename_moves_bound_symbol_fields_too():
    """Delay.shift and Algebraic.lhs are Symbol fields, not just the rhs."""
    renamed = sem.rename(sem.SEMANTICS["Delay"].equations[0], "d1")
    assert (renamed.output, renamed.input, renamed.shift) == (
        sem.qualify(sem.y, "d1"),
        sem.qualify(sem.x, "d1"),
        sem.qualify(sem.delay, "d1"),
    )
    assert renamed.shift.is_positive


def test_rename_preserves_the_equation_under_back_substitution():
    """Renaming is a relabelling, not a rewrite."""
    original = sem.SEMANTICS["LI"].equations[0]
    renamed = sem.rename(original, "n1")
    back = {sem.qualify(s, "n1"): s for s in original.rhs.free_symbols}
    assert sp.simplify(renamed.rhs.xreplace(back) - original.rhs) == 0


def test_rename_leaves_the_table_untouched():
    before = sem.SEMANTICS["LI"].equations[0]
    sem.rename(before, "n1")
    assert sem.SEMANTICS["LI"].equations[0] is before
    assert before.state == sem.v


def test_rename_does_not_scope_time():
    """Time is shared by the whole graph; a per-node t would desynchronise it."""
    with_time = sem.Derivative(sem.v, sem.v * sem.t)
    renamed = sem.rename(with_time, "n1")
    assert sem.t in renamed.rhs.free_symbols
    assert sem.qualify(sem.t, "n1") not in renamed.rhs.free_symbols


def test_two_nodes_renamed_apart_no_longer_alias():
    """The bug this machinery exists for, at the form level."""
    first = sem.rename(sem.SEMANTICS["LI"].equations[0], "n1")
    second = sem.rename(sem.SEMANTICS["LI"].equations[0], "n2")
    assert first != second
    assert first.rhs.free_symbols & second.rhs.free_symbols == set()


def test_rename_distinguishes_a_transition_from_its_neighbour():
    transition = sem.rename(sem.SEMANTICS["Threshold"].equations[0], "th")
    flow = sem.rename(sem.SEMANTICS["LI"].equations[0], "li")
    assert transition.arg.free_symbols & flow.rhs.free_symbols == set()


@pytest.mark.xfail(
    strict=True, reason="sem.system() is a TODO; remove marker when written"
)
def test_graph_composes_into_one_system_with_disjoint_states():
    """Two LI nodes in one graph must end up with distinct state symbols.

    Written against the property rather than the design, so it holds for any of
    the three scoping options. Delete the marker when `system` lands.
    """
    graph = nir.NIRGraph(nodes={"a": mock_li(2), "b": mock_li(2)}, edges=[("a", "b")])
    equations = sem.system(sem.from_nir(graph))
    states = [eq.state for eq in equations if isinstance(eq, sem.Derivative)]
    assert len(states) == len(set(states)) == 2


def test_ports_separate_inputs_from_parameters():
    """`v` is a state in LI and an input in Threshold, so the classification
    cannot come from the symbol."""
    li = sem.ports(sem.from_nir(nir.LI(tau=ONE, r=ONE, v_leak=ONE)))
    assert li == ((sem.i,), sem.v)
    assert sem.tau not in li[0] and sem.v_leak not in li[0]

    threshold = sem.ports(sem.from_nir(nir.Threshold(threshold=ONE)))
    assert threshold == ((sem.v,), sem.z)


def test_ports_hides_a_sibling_bound_symbol_and_outputs_the_spike():
    """IF's z is defined by its own Crossing, so no edge may bind it, and the
    node emits the spike rather than the membrane."""
    inputs, output = sem.ports(sem.from_nir(nir.IF(r=ONE, v_threshold=ONE)))
    assert inputs == (sem.i,)
    assert sem.z not in inputs
    assert output == sem.z


NODES = {
    "I": nir.I(r=ONE),
    "LI": nir.LI(tau=ONE, r=ONE, v_leak=ONE),
    "Linear": nir.Linear(weight=np.eye(2)),
    "Scale": nir.Scale(scale=ONE),
    "Delay": nir.Delay(delay=ONE),
    "Threshold": nir.Threshold(threshold=ONE),
    "IF": nir.IF(r=ONE, v_threshold=ONE),
    "LIF": nir.LIF(tau=ONE, r=ONE, v_leak=ONE, v_threshold=ONE),
}


def test_every_table_entry_has_an_instance_to_test_against():
    assert set(NODES) == set(sem.SEMANTICS)


@pytest.mark.parametrize("name", sorted(NODES))
def test_every_parameter_symbol_binds_to_a_real_field(name):
    """The alias table is the only hand-written half of the binding, so it is
    the half that can drift from nir.ir. Whatever `ports` does not call an input
    and no equation defines is a parameter, and must name a field."""
    node_sem = sem.SEMANTICS[name]
    declared = {f.name for f in dataclasses.fields(getattr(nir, name))}
    inputs, _output = sem.ports(sem.from_nir(NODES[name]))
    defined = {e.defines for e in node_sem.equations}
    free = set().union(*(sem._free(e) for e in node_sem.equations))
    for symbol in free - defined - set(inputs):
        assert node_sem.field_of(symbol) in declared, (name, symbol)


def test_guarded_jump_is_not_yet_constructible():
    """Candidate (A) vs (B) is unsettled, so the merge refuses to pretend."""
    with pytest.raises(NotImplementedError):
        sem.GuardedJump()


# --- claims not yet established ----------------------------------------------


@pytest.mark.skip(
    reason="TODO: the existence argument is unverified; see OPEN in semantics.py"
)
def test_subtract_theta_can_strand_the_state():
    """The third leg of the reset argument, which is so far an argument.

    Scenario to build:
      * LI with tau, r, v_leak = 0, threshold theta;
      * an impulsive input at t0 driving v to 3*theta;
      * subtract-theta: v lands at 2*theta, still in the jump set, so the guard
        holds while a second jump at the same t is forbidden -- the state can
        neither flow nor jump;
      * reset-to-v_reset with v_reset < theta: always leaves the jump set, so at
        most one jump per t holds for free.
    Assert the stranding for subtract-theta and its absence for reset-to.
    """
