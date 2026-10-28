# Copyright 2026 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections

import numpy as np
import pytest
import sympy

import cirq


def _record_distribution(
    branches: list[cirq.CircuitBranch], key: str = "m"
) -> dict[tuple[int, ...], float]:
    """Sums branch probabilities over recorded digit tuples for `key`."""
    result: dict[tuple[int, ...], float] = collections.defaultdict(float)
    for branch in branches:
        for record in branch.measurements[key]:
            result[record] += branch.probability
    return dict(result)


def test_empty_circuit_is_one_certain_branch() -> None:
    branches = cirq.enumerate_branches(cirq.Circuit())
    assert len(branches) == 1
    assert branches[0].probability == 1.0
    assert branches[0].measurements == {}
    assert branches[0].channel_measurements == {}


def test_purely_unitary_circuit_is_one_branch() -> None:
    a, b = cirq.LineQubit.range(2)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.H(b)))
    assert len(branches) == 1
    assert branches[0].probability == 1.0
    assert np.isclose(np.linalg.norm(branches[0].final_state_vector), 1.0)


def test_simple_measurement_branches_in_order() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.H(a), cirq.measure(a, key="m")))
    assert len(branches) == 2
    assert [b.measurements["m"] for b in branches] == [((0,),), ((1,),)]
    assert [b.probability for b in branches] == [pytest.approx(0.5)] * 2
    assert np.allclose(branches[0].final_state_vector, [1, 0])
    assert np.allclose(branches[1].final_state_vector, [0, 1])
    assert all(np.isclose(np.linalg.norm(b.final_state_vector), 1.0) for b in branches)
    assert sum(b.probability for b in branches) == pytest.approx(1.0)


def test_bell_measurement() -> None:
    a, b = cirq.LineQubit.range(2)
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.measure(a, b, key="m"))
    )
    assert [b.measurements["m"] for b in branches] == [((0, 0),), ((1, 1),)]
    assert np.allclose(branches[0].final_state_vector, [1, 0, 0, 0])
    assert np.allclose(branches[1].final_state_vector, [0, 0, 0, 1])


def test_probabilities_are_squared_amplitudes() -> None:
    a = cirq.LineQubit(0)
    initial_state = np.array([0.6, 0.8])
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.measure(a, key="m")), initial_state=initial_state
    )
    assert branches[0].probability == pytest.approx(0.36)
    assert branches[1].probability == pytest.approx(0.64)


def test_qubit_order_controls_state_vector_order() -> None:
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(b), cirq.measure(a, b, key="m"))
    branches = cirq.enumerate_branches(circuit)
    assert np.allclose(branches[0].final_state_vector, [1, 0, 0, 0])
    assert np.allclose(branches[1].final_state_vector, [0, 1, 0, 0])

    reversed_branches = cirq.enumerate_branches(circuit, qubit_order=[b, a])
    assert np.allclose(reversed_branches[0].final_state_vector, [1, 0, 0, 0])
    assert np.allclose(reversed_branches[1].final_state_vector, [0, 0, 1, 0])


def test_mid_circuit_measurement_drives_if() -> None:
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.measure(a, key="m"),
        cirq.If(cirq.KeyCondition(cirq.MeasurementKey("m")), cirq.X(b)),
    )
    branches = cirq.enumerate_branches(circuit)
    assert [b.measurements["m"] for b in branches] == [((0,),), ((1,),)]
    assert np.allclose(branches[0].final_state_vector, [1, 0, 0, 0])
    assert np.allclose(branches[1].final_state_vector, [0, 0, 0, 1])


def test_mid_circuit_measurement_drives_with_classical_controls() -> None:
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.measure(a, key="m"),
        cirq.X(b).with_classical_controls("m"),
    )
    branches = cirq.enumerate_branches(circuit)
    assert np.allclose(branches[0].final_state_vector, [1, 0, 0, 0])
    assert np.allclose(branches[1].final_state_vector, [0, 0, 0, 1])


def test_confusion_map_on_definite_state() -> None:
    a = cirq.LineQubit(0)
    confusion = np.array([[0.8, 0.2], [0.2, 0.8]])
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.measure(a, key="m", confusion_map={(0,): confusion}))
    )
    distribution = _record_distribution(branches)
    assert distribution[(0,)] == pytest.approx(0.8)
    assert distribution[(1,)] == pytest.approx(0.2)
    # Readout noise only changes the classical record, not the quantum state.
    assert all(np.allclose(b.final_state_vector, [1, 0]) for b in branches)
    assert sum(b.probability for b in branches) == pytest.approx(1.0)


def test_confusion_map_with_true_branching() -> None:
    a = cirq.LineQubit(0)
    confusion = np.array([[0.8, 0.2], [0.2, 0.8]])
    branches = cirq.enumerate_branches(
        cirq.Circuit(
            cirq.H(a),
            cirq.measure(a, key="m", confusion_map={(0,): confusion}),
        )
    )
    distribution = _record_distribution(branches)
    # Both true outcomes are equally likely and have the same confusion row.
    assert distribution[(0,)] == pytest.approx(0.5)
    assert distribution[(1,)] == pytest.approx(0.5)


def test_confusion_map_multiple_qubit_group() -> None:
    a, b = cirq.LineQubit.range(2)
    confusion = np.full((4, 4), 0.25)
    branches = cirq.enumerate_branches(
        cirq.Circuit(
            cirq.H(a),
            cirq.measure(a, b, key="m", confusion_map={(0, 1): confusion}),
        )
    )
    assert len(branches) == 8
    assert sorted(b.probability for b in branches) == [pytest.approx(0.125)] * 8


def test_confusion_map_overlapping_groups_matches_act_on_semantics() -> None:
    # `SimulationState._confuse_result` selects every confuser row from the
    # original (unconfused) bits while later groups overwrite shared indices.
    a, b = cirq.LineQubit.range(2)
    confusion_ab = np.array([
        [0.5, 0.1, 0.2, 0.2],
        [0.1, 0.5, 0.2, 0.2],
        [0.3, 0.3, 0.2, 0.2],
        [0.2, 0.2, 0.3, 0.3],
    ])
    confusion_a = np.array([[0.6, 0.4], [0.25, 0.75]])
    circuit = cirq.Circuit(
        cirq.measure(
            a,
            b,
            key="m",
            confusion_map={(0, 1): confusion_ab, (0,): confusion_a},
        ),
        # Force the mid-circuit (act-on) measurement path in the simulator.
        cirq.H(a),
    )
    branches = cirq.enumerate_branches(circuit)
    distribution = _record_distribution(branches)
    assert distribution[(0, 0)] == pytest.approx(0.42)
    assert distribution[(0, 1)] == pytest.approx(0.18)
    assert distribution[(1, 0)] == pytest.approx(0.28)
    assert distribution[(1, 1)] == pytest.approx(0.12)


def test_invert_mask_flips_recorded_bit() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.measure(a, key="m", invert_mask=(True,))))
    assert branches[0].measurements["m"] == ((1,),)
    assert np.allclose(branches[0].final_state_vector, [1, 0])


def test_confusion_is_applied_before_invert_mask() -> None:
    a = cirq.LineQubit(0)
    confusion = np.array([[0.8, 0.2], [0.2, 0.8]])
    branches = cirq.enumerate_branches(
        cirq.Circuit(
            cirq.measure(
                a,
                key="m",
                invert_mask=(True,),
                confusion_map={(0,): confusion},
            )
        )
    )
    distribution = _record_distribution(branches)
    # Confuse first (0 -> 0 with .8, 1 with .2), then invert the record.
    assert distribution[(1,)] == pytest.approx(0.8)
    assert distribution[(0,)] == pytest.approx(0.2)


def test_qutrit_measurement() -> None:
    q = cirq.LineQid(0, dimension=3)
    dft = cirq.MatrixGate(np.fft.fft(np.eye(3)) / np.sqrt(3), qid_shape=(3,))
    branches = cirq.enumerate_branches(cirq.Circuit(dft.on(q), cirq.measure(q, key="m")))
    assert len(branches) == 3
    assert [b.probability for b in branches] == [pytest.approx(1 / 3)] * 3
    assert [b.measurements["m"] for b in branches] == [
        ((0,),),
        ((1,),),
        ((2,),),
    ]
    for i, branch in enumerate(branches):
        assert np.allclose(branch.final_state_vector, np.eye(3)[i])


def test_qutrit_confusion_map() -> None:
    q = cirq.LineQid(0, dimension=3)
    confusion = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.measure(q, key="m", confusion_map={(0,): confusion}))
    )
    assert len(branches) == 1
    assert branches[0].measurements["m"] == ((1,),)
    assert np.allclose(branches[0].final_state_vector, [1, 0, 0])


def test_repeated_measurement_key_keeps_full_history() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(
        cirq.Circuit(
            cirq.H(a),
            cirq.measure(a, key="m"),
            cirq.H(a),
            cirq.measure(a, key="m"),
        )
    )
    assert len(branches) == 4
    assert [b.measurements["m"] for b in branches] == [
        ((0,), (0,)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((1,), (1,)),
    ]
    assert sorted(b.probability for b in branches) == [pytest.approx(0.25)] * 4
    # Sibling histories must not share record lists: branches sharing the
    # first outcome still carry independent second outcomes ((0, 0) vs (0, 1)).
    assert branches[0].measurements["m"][1] == (0,)
    assert branches[1].measurements["m"][1] == (1,)


def test_random_gate_channel_branches_over_mixture() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.H(a), cirq.Z.with_probability(0.3).on(a)))
    assert len(branches) == 2
    assert sorted(b.probability for b in branches) == [
        pytest.approx(0.3),
        pytest.approx(0.7),
    ]
    # Mixture index 0 is Z, index 1 is the identity.
    assert np.allclose(branches[0].final_state_vector, [1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert np.allclose(branches[1].final_state_vector, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_random_gate_channel_converging_branches_are_merged() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.H(a), cirq.X.with_probability(0.5).on(a)))
    # X|+> = |+>, so both mixture components converge to the same state.
    assert len(branches) == 1
    assert branches[0].probability == pytest.approx(1.0)
    assert np.allclose(branches[0].final_state_vector, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_keyed_mixed_unitary_channel_records_index() -> None:
    a = cirq.LineQubit(0)
    keyed = cirq.MixedUnitaryChannel([(0.3, cirq.unitary(cirq.X)), (0.7, np.eye(2))], key="k").on(a)
    branches = cirq.enumerate_branches(cirq.Circuit(keyed, cirq.measure(a, key="m")))
    assert len(branches) == 2
    # Index 0 applies X (state |1>, measured 1) with probability .3.
    assert branches[0].channel_measurements["k"] == (0,)
    assert branches[0].measurements["m"] == ((1,),)
    assert branches[0].probability == pytest.approx(0.3)
    # Index 1 is the identity (state |0>, measured 0) with probability .7.
    assert branches[1].channel_measurements["k"] == (1,)
    assert branches[1].measurements["m"] == ((0,),)
    assert branches[1].probability == pytest.approx(0.7)


def test_channel_record_drives_classical_control() -> None:
    a, b = cirq.LineQubit.range(2)
    keyed = cirq.MixedUnitaryChannel([(0.3, cirq.unitary(cirq.X)), (0.7, np.eye(2))], key="k").on(a)
    circuit = cirq.Circuit(
        keyed,
        cirq.If(cirq.KeyCondition(cirq.MeasurementKey("k")), cirq.X(b)),
        cirq.measure(a, key="m"),
        cirq.measure(b, key="n"),
    )
    branches = cirq.enumerate_branches(circuit)
    joint = {
        (
            b.channel_measurements["k"][0],
            b.measurements["m"][0][0],
            b.measurements["n"][0][0],
        ): b.probability
        for b in branches
    }
    # k=0 means X was applied (a=1); a zero index disables the feedback.
    assert joint[(0, 1, 0)] == pytest.approx(0.3)
    # k=1 means identity (a=0); non-zero index enables X on b.
    assert joint[(1, 0, 1)] == pytest.approx(0.7)


def test_repeated_keyed_channel_records_every_index() -> None:
    a = cirq.LineQubit(0)
    keyed = cirq.MixedUnitaryChannel([(0.5, cirq.unitary(cirq.X)), (0.5, np.eye(2))], key="k").on(a)
    branches = cirq.enumerate_branches(cirq.Circuit(keyed, keyed, cirq.measure(a, key="m")))
    assert len(branches) == 4
    assert all(len(b.channel_measurements["k"]) == 2 for b in branches)
    histories = {b.channel_measurements["k"] for b in branches}
    assert histories == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert sum(b.probability for b in branches) == pytest.approx(1.0)


def test_reset_collapses_to_single_branch() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.H(a), cirq.reset(a)))
    assert len(branches) == 1
    assert branches[0].probability == pytest.approx(1.0)
    assert np.allclose(branches[0].final_state_vector, [1, 0])
    # Reset is unobserved and records nothing.
    assert branches[0].measurements == {}


def test_qutrit_reset() -> None:
    q = cirq.LineQid(0, dimension=3)
    dft = cirq.MatrixGate(np.fft.fft(np.eye(3)) / np.sqrt(3), qid_shape=(3,))
    branches = cirq.enumerate_branches(cirq.Circuit(dft.on(q), cirq.reset(q)))
    assert len(branches) == 1
    assert branches[0].probability == pytest.approx(1.0)
    assert np.allclose(branches[0].final_state_vector, [1, 0, 0])


def test_static_circuit_operation_repetitions() -> None:
    a = cirq.LineQubit(0)
    inner = cirq.FrozenCircuit(cirq.H(a), cirq.measure(a, key="m"))
    circuit = cirq.Circuit(cirq.CircuitOperation(inner, repetitions=2))
    branches = cirq.enumerate_branches(circuit)
    assert len(branches) == 4
    assert all(len(b.measurements["m"]) == 2 for b in branches)
    assert sorted(b.probability for b in branches) == [pytest.approx(0.25)] * 4


def test_circuit_operation_qubit_and_key_maps() -> None:
    a = cirq.LineQubit(0)
    x = cirq.NamedQubit("x")
    inner = cirq.FrozenCircuit(cirq.H(a), cirq.measure(a, key="m"))
    mapped = cirq.CircuitOperation(inner, qubit_map={a: x}, measurement_key_map={"m": "k"})
    branches = cirq.enumerate_branches(cirq.Circuit(mapped), qubit_order=[x])
    assert len(branches) == 2
    assert all(set(b.measurements) == {"k"} for b in branches)
    assert all("m" not in b.measurements for b in branches)


def test_pauli_measurement_gate_decomposes() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.H(a), cirq.PauliMeasurementGate([cirq.X], key="m").on(a))
    )
    # |+> measured along X deterministically yields +1, recorded as 0.
    assert len(branches) == 1
    assert branches[0].measurements["m"] == ((0,),)
    assert np.allclose(branches[0].final_state_vector, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_set_variable_resolves_later_gate() -> None:
    a = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.SetVariable(sympy.Symbol("t"), 0.5),
        (cirq.X ** sympy.Symbol("t"))(a),
    )
    branches = cirq.enumerate_branches(circuit)
    assert len(branches) == 1
    assert np.allclose(branches[0].final_state_vector, [0.5 + 0.5j, 0.5 - 0.5j])


def test_param_resolver_argument() -> None:
    a = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X ** sympy.Symbol("t"))(a))
    branches = cirq.enumerate_branches(circuit, param_resolver={"t": 0})
    assert np.allclose(branches[0].final_state_vector, [1, 0])


def test_op_tree_input_is_accepted() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches([cirq.H(a), cirq.measure(a, key="m")])
    assert len(branches) == 2


def test_atol_drops_low_probability_outcomes() -> None:
    a = cirq.LineQubit(0)
    confusion = np.array([[1 - 1e-10, 1e-10], [0.0, 1.0]])
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.measure(a, key="m", confusion_map={(0,): confusion})),
        atol=1e-8,
    )
    assert len(branches) == 1
    assert branches[0].measurements["m"] == ((0,),)


def test_complex64_dtype() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(
        cirq.Circuit(cirq.H(a), cirq.measure(a, key="m")),
        dtype=np.complex64,
    )
    assert branches[0].final_state_vector.dtype == np.complex64


def test_returned_state_vectors_are_independent() -> None:
    a = cirq.LineQubit(0)
    branches = cirq.enumerate_branches(cirq.Circuit(cirq.H(a), cirq.measure(a, key="m")))
    other_vector = branches[1].final_state_vector.copy()
    branches[0].final_state_vector[:] = 0
    assert np.allclose(branches[1].final_state_vector, other_vector)


def test_stable_order_across_calls() -> None:
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.H(b),
        cirq.measure(a, key="x"),
        cirq.measure(b, key="y"),
    )
    first = cirq.enumerate_branches(circuit)
    second = cirq.enumerate_branches(circuit)
    expected_histories = [
        {"x": ((0,),), "y": ((0,),)},
        {"x": ((0,),), "y": ((1,),)},
        {"x": ((1,),), "y": ((0,),)},
        {"x": ((1,),), "y": ((1,),)},
    ]
    assert [b.measurements for b in first] == expected_histories
    assert [b.measurements for b in first] == [b.measurements for b in second]


def test_agrees_with_simulator_sampling() -> None:
    a, b = cirq.LineQubit.range(2)
    confusion = np.array([[0.8, 0.2], [0.2, 0.8]])
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.measure(a, key="m", confusion_map={(0,): confusion}),
        cirq.X(b).with_classical_controls("m"),
        cirq.measure(b, key="n"),
    )
    branches = cirq.enumerate_branches(circuit)
    enum_distribution: dict[tuple[int, int], float] = collections.defaultdict(float)
    for branch in branches:
        key = (branch.measurements["m"][0][0], branch.measurements["n"][0][0])
        enum_distribution[key] += branch.probability

    result = cirq.Simulator(seed=1234).run(circuit, repetitions=10000)
    sampled = collections.Counter(
        zip(
            result.records["m"].reshape(-1).tolist(),
            result.records["n"].reshape(-1).tolist(),
        )
    )
    assert set(sampled) == set(enum_distribution)
    for key, expected in enum_distribution.items():
        assert sampled[key] / 10000 == pytest.approx(expected, abs=0.03)


def test_while_loop_raises() -> None:
    a = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.While(cirq.KeyCondition(cirq.MeasurementKey("m")), cirq.X(a)))
    with pytest.raises(ValueError, match="loop"):
        cirq.enumerate_branches(circuit)


def test_repeat_until_circuit_operation_raises() -> None:
    a = cirq.LineQubit(0)
    inner = cirq.FrozenCircuit(cirq.measure(a, key="m"))
    operation = cirq.CircuitOperation(
        inner, repeat_until=cirq.KeyCondition(cirq.MeasurementKey("m"))
    )
    with pytest.raises(ValueError, match="loop"):
        cirq.enumerate_branches(cirq.Circuit(operation))


def test_general_kraus_channel_raises_type_error() -> None:
    a = cirq.LineQubit(0)
    with pytest.raises(TypeError, match="pure state"):
        cirq.enumerate_branches(cirq.Circuit(cirq.amplitude_damp(0.5).on(a)))


def test_unresolved_parameter_raises() -> None:
    a = cirq.LineQubit(0)
    with pytest.raises(ValueError, match="parameter sweep"):
        cirq.enumerate_branches(cirq.Circuit((cirq.X ** sympy.Symbol("t"))(a)))


def test_classical_control_on_missing_key_raises() -> None:
    a = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(a).with_classical_controls("missing"))
    with pytest.raises((ValueError, KeyError)):
        cirq.enumerate_branches(circuit)


def test_alias_is_same_function() -> None:
    assert cirq.enumerate_dynamic_circuit_branches is cirq.enumerate_branches
    assert cirq.sim.CircuitBranch is cirq.CircuitBranch


def test_circuit_branch_repr() -> None:
    a = cirq.LineQubit(0)
    branch = cirq.enumerate_branches(cirq.Circuit(cirq.H(a)))[0]
    text = repr(branch)
    assert text.startswith("cirq.CircuitBranch(")
    assert "probability=1.0" in text
