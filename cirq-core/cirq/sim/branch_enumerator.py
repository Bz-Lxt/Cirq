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

"""Exact enumeration of all probability branches of loop-free dynamic circuits.

This module is a debugging aid for dynamic circuits that contain mid-circuit
measurements (possibly with readout confusion matrices), random (mixture)
channels such as `cirq.RandomGateChannel`, and classical feedback such as
`cirq.If` / `Operation.with_classical_controls`.

Instead of sampling, `enumerate_branches` walks the circuit once, splitting a
pure state vector into a new branch for *every* possible non-zero-probability
outcome of each stochastic operation. Each returned `CircuitBranch` carries the
joint probability of the whole branch, the complete measurement key history
(including repeated measurements on the same key and keyed channel indices),
and the normalized post-measurement state vector. Branches are returned in a
stable lexicographic order by their chronological outcome path, and branches
that differ only by unobserved choices that converge to the same state and
classical history (e.g. `cirq.ResetChannel`) are merged by summing their
probabilities.

Only operations that can be evaluated exactly on a pure state vector are
accepted: unitaries, mixtures, computational-basis measurements, resets,
classical controls, (static) circuit operations and `cirq.SetVariable`.
Loops (`cirq.While`, `repeat_until` circuit operations), unresolved parameters,
and general (non-mixture) Kraus channels raise an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast, TYPE_CHECKING

import attrs
import numpy as np

from cirq import circuits, linalg, ops, protocols, study, value
from cirq.ops import common_gates
from cirq.sim import simulation_utils
from cirq.sim.state_vector_simulation_state import StateVectorSimulationState

if TYPE_CHECKING:
    import cirq


@attrs.frozen
class CircuitBranch:
    """One non-zero-probability branch of an enumerated dynamic circuit.

    Attributes:
        probability: The joint probability of every stochastic outcome that
            occurred along this branch. The probabilities of all returned
            branches sum to one (up to floating-point error).
        classical_data: The complete classical data of the branch. Holds the
            full per-key measurement history, including repeated measurements
            on the same key (records are appended in order) as well as channel
            measurement indices. Use `classical_data.records`,
            `classical_data.channel_records` and `classical_data.keys()`.
        final_state_vector: The normalized final state vector as a one
            dimensional NumPy array, in the canonical Kronecker order of the
            qubits supplied to `enumerate_branches`.
    """

    probability: float
    classical_data: cirq.ClassicalDataStoreReader
    final_state_vector: np.ndarray = attrs.field(eq=False)

    @property
    def measurements(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        """The computational-basis measurement history keyed by measurement key.

        Maps each measurement key to the tuple of digit tuples recorded for
        that key, in chronological order (repeated keys produce multiple
        entries).
        """
        return {str(k): tuple(v) for k, v in self.classical_data.records.items()}

    @property
    def channel_measurements(self) -> dict[str, tuple[int, ...]]:
        """The keyed random-channel outcome history keyed by measurement key.

        Maps each keyed random channel to the tuple of mixture indices that
        were selected for it, in chronological order.
        """
        return {str(k): tuple(v) for k, v in self.classical_data.channel_records.items()}

    def __repr__(self) -> str:
        return (
            f"cirq.CircuitBranch(probability={self.probability!r}, "
            f"classical_data={self.classical_data!r})"
        )


@attrs.frozen
class _PendingBranch:
    """An in-flight branch while the circuit is being walked."""

    probability: float
    state: StateVectorSimulationState
    # Chronological sequence of outcome indices (measurement outcomes,
    # confusion outcomes, mixture indices, ...) that led to this branch. Used
    # for stable ordering and merging.
    path: tuple[int, ...]


def enumerate_branches(
    program: cirq.AbstractCircuit | cirq.OP_TREE,
    *,
    initial_state: cirq.STATE_VECTOR_LIKE = 0,
    qubit_order: cirq.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    param_resolver: cirq.ParamResolverOrSimilarType = None,
    dtype: type[np.complexfloating] = np.complex128,
    atol: float = 0.0,
) -> list[CircuitBranch]:
    """Enumerates all non-zero-probability branches of a loop-free dynamic circuit.

    The circuit is interpreted exactly as by Cirq's state vector simulator,
    except that instead of drawing one random outcome at each stochastic
    operation, a separate branch is created for every outcome whose probability
    is greater than `atol`. Mid-circuit measurements collapse the state of the
    branch and record (optionally confused and inverted) digits under their
    measurement key; mixture (random) channels branch over their components,
    recording the selected component index when the channel has a measurement
    key; classical controls are resolved independently within each branch.

    Args:
        program: The circuit (or op tree) to enumerate. It must not contain
            loops.
        initial_state: The initial pure state. If an integer, it is the
            computational basis state. If a NumPy array, it is the full state
            vector in the order given by `qubit_order`.
        qubit_order: Determines the canonical ordering of the qubits and hence
            the basis order of the returned state vectors.
        param_resolver: Parameters to run with the program. Parameters may also
            be assigned during execution by `cirq.SetVariable`.
        dtype: The complex dtype of the simulated state vectors. Defaults to
            `numpy.complex128` for accurate probabilities.
        atol: Outcomes with probability less than or equal to this value are
            treated as impossible and omitted. Confusion matrix entries and
            mixture components are thresholded the same way. Defaults to zero,
            i.e. only exactly impossible outcomes are omitted.

    Returns:
        A list of `CircuitBranch` in stable lexicographic order of the
        branches' chronological outcome paths. The joint probabilities sum to
        one and each final state vector is normalized.

    Raises:
        ValueError: If the circuit contains a loop (`cirq.While` or a
            `cirq.CircuitOperation` with `repeat_until`), an unresolved
            parameter, or a repeated measurement key whose shape does not
            match.
        TypeError: If the circuit contains an operation that cannot be
            evaluated exactly on a pure state vector (e.g. a general
            non-mixture Kraus channel), or if a classical control references a
            measurement key that has not been recorded.
    """
    if isinstance(program, circuits.AbstractCircuit):
        circuit = program
    else:
        circuit = circuits.Circuit(program)
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        q for q in circuit.all_qubits() if not isinstance(q, ops.VariableQid)
    )
    if param_resolver is not None:
        resolver = study.ParamResolver(param_resolver)
    else:
        resolver = study.ParamResolver({})
    state = StateVectorSimulationState(
        qubits=qubits,
        initial_state=initial_state,
        dtype=dtype,
        classical_data=value.ClassicalDataDictionaryStore(),
        param_resolver=resolver,
    )
    branches = [_PendingBranch(probability=1.0, state=state, path=())]
    for op in circuit.all_operations():
        new_branches: list[_PendingBranch] = []
        for branch in branches:
            new_branches.extend(_apply_op(op, branch, atol))
        branches = new_branches

    return _finalize_branches(branches)


def _fork_state(state: StateVectorSimulationState) -> StateVectorSimulationState:
    """Copies a branch state, including an independent classical-data store.

    `ClassicalDataDictionaryStore.copy` only shallow-copies its dictionaries, so
    the per-key record lists would otherwise be shared between branches and a
    repeated measurement on one branch would corrupt its siblings.
    """
    new_state = state.copy()
    old = cast("cirq.ClassicalDataDictionaryStore", new_state._classical_data)
    new_store = value.ClassicalDataDictionaryStore(
        _records={key: list(records) for key, records in old.records.items()},
        _measured_qubits={key: list(measured) for key, measured in old.measured_qubits.items()},
        _channel_records={key: list(records) for key, records in old.channel_records.items()},
        _measurement_types=dict(old.measurement_types),
    )
    new_state._classical_data = new_store
    return new_state


def _resolve_op(op: cirq.Operation, branch: _PendingBranch) -> cirq.Operation:
    """Resolves parameters on `op` using the (possibly updated) branch resolver."""
    raw = getattr(op, "untagged", op)
    if isinstance(raw, ops.SetVariable):
        # SetVariable reads and writes the resolver itself at execution time.
        return op
    if protocols.is_parameterized(op):
        op = protocols.resolve_parameters(op, branch.state.param_resolver)
        if protocols.is_parameterized(op):
            raise ValueError(
                "Circuit contains ops whose symbols were not specified in the "
                f"parameter sweep. Ops: [{op!r}]"
            )
    return op


def _apply_op(op: cirq.Operation, branch: _PendingBranch, atol: float) -> list[_PendingBranch]:
    """Applies one operation, returning zero or more resulting branches."""
    op = _resolve_op(op, branch)
    raw = getattr(op, "untagged", op)

    if isinstance(raw, ops.While):
        raise ValueError(
            "Cannot exactly enumerate branches of circuits containing loops: "
            f"{raw!r}. Loops can have an unbounded number of executions."
        )

    if isinstance(raw, ops.SetVariable):
        protocols.act_on(op, branch.state)
        return [branch]

    if isinstance(raw, (ops.If, ops.ClassicallyControlledOperation)):
        if isinstance(raw, ops.If):
            conditions = raw.conditions
            sub_operation = raw.sub_operation
        else:
            conditions = raw._conditions
            sub_operation = raw._sub_operation
        if all(c.resolve(branch.state.classical_data) for c in conditions):
            return _apply_op(sub_operation, branch, atol)
        return [branch]

    if isinstance(raw, circuits.CircuitOperation):
        if raw.repeat_until is not None:
            raise ValueError(
                "Cannot exactly enumerate branches of circuits containing loops: "
                f"{raw!r} has a `repeat_until` condition."
            )
        # `mapped_circuit` applies repetitions, qubit/key maps and parameter
        # overrides, matching `CircuitOperation._act_on_` for the static case.
        branches = [branch]
        for inner_op in raw.mapped_circuit(deep=False).all_operations():
            branches = [
                out for inner_branch in branches for out in _apply_op(inner_op, inner_branch, atol)
            ]
        return branches

    gate = getattr(raw, "gate", None)
    if isinstance(gate, ops.MeasurementGate):
        return _measurement_branches(raw, gate, branch, atol)
    if isinstance(gate, ops.ResetChannel):
        return _reset_branches(raw, gate, branch, atol)

    if protocols.has_unitary(raw):
        protocols.act_on(op, branch.state)
        return [branch]

    if protocols.has_mixture(raw):
        return _mixture_branches(raw, branch, atol)

    return _decompose_branches(raw, branch, atol)


def _measurement_branches(
    op: cirq.Operation,
    gate: cirq.MeasurementGate,
    branch: _PendingBranch,
    atol: float,
) -> list[_PendingBranch]:
    """Branches over true outcomes, confusion outcomes and recorded digits."""
    state = branch.state
    qubits = op.qubits
    axes = state.get_axes(qubits)
    tensor = state.target_tensor
    full_shape = tensor.shape
    probabilities = simulation_utils.state_probabilities_by_indices(
        (tensor * tensor.conj()).real, axes, full_shape
    )
    meas_shape = tuple(full_shape[axis] for axis in axes)
    invert_mask = gate.full_invert_mask()

    result: list[_PendingBranch] = []
    for true_value, true_prob in enumerate(probabilities):
        if true_prob <= atol:
            continue
        true_digits = value.big_endian_int_to_digits(true_value, base=meas_shape)

        # Readout noise: each confusion matrix is applied sequentially and can
        # itself split into several recorded outcomes. Only the classical record
        # changes; the quantum state has already collapsed to `true_digits`.
        # Mirror `SimulationState._confuse_result`: each confuser's row is
        # selected from the original (unconfused) `true_digits`, while its
        # output digits overwrite the running record (later groups win for
        # overlapping indices).
        variants: list[tuple[list[int], float]] = [(list(true_digits), 1.0)]
        for indices, confuser in gate.confusion_map.items():
            confused_variants: list[tuple[list[int], float]] = []
            confusion_matrix = np.asarray(confuser)
            mat_dims = [qubits[k].dimension for k in indices]
            value_count = confusion_matrix.shape[1]
            confusion_row = value.big_endian_digits_to_int(
                (true_digits[k] for k in indices), base=mat_dims
            )
            for current_digits, confusion_weight in variants:
                for new_value in range(value_count):
                    conditional_prob = float(confusion_matrix[confusion_row, new_value])
                    if conditional_prob <= atol:
                        continue
                    new_digits = current_digits.copy()
                    new_sub_digits = value.big_endian_int_to_digits(new_value, base=mat_dims)
                    for i, k in enumerate(indices):
                        new_digits[k] = new_sub_digits[i]
                    confused_variants.append((new_digits, confusion_weight * conditional_prob))
            variants = confused_variants

        for confused_digits, confusion_weight in variants:
            # Match `SimulationState.measure`: inversion is only defined for
            # binary (digit < 2) outcomes.
            recorded_digits = tuple(
                digit ^ (1 if digit < 2 and bool(mask) else 0)
                for digit, mask in zip(confused_digits, invert_mask)
            )

            new_state = _fork_state(state)
            new_tensor = new_state.target_tensor
            result_slice = linalg.slice_for_qubits_equal_to(
                axes, big_endian_qureg_value=true_value, qid_shape=full_shape
            )
            mask_array = np.ones(full_shape, dtype=bool)
            mask_array[result_slice] = False
            new_tensor[mask_array] = 0
            new_tensor /= np.sqrt(true_prob)

            new_state._classical_data.record_measurement(gate.mkey, recorded_digits, qubits)
            recorded_value = value.big_endian_digits_to_int(recorded_digits, base=list(meas_shape))
            result.append(
                _PendingBranch(
                    probability=branch.probability * float(true_prob) * confusion_weight,
                    state=new_state,
                    path=(*branch.path, recorded_value),
                )
            )
    return result


def _reset_branches(
    op: cirq.Operation,
    gate: cirq.ResetChannel,
    branch: _PendingBranch,
    atol: float,
) -> list[_PendingBranch]:
    """Unobserved projective measurement followed by a flip back to ``|0⟩``.

    Mirrors `ResetChannel._act_on_` for pure states, enumerating every outcome.
    All outcomes converge to the same state, so the branches merge later.
    """
    state = branch.state
    qubits = op.qubits
    axes = state.get_axes(qubits)
    tensor = state.target_tensor
    full_shape = tensor.shape
    probabilities = simulation_utils.state_probabilities_by_indices(
        (tensor * tensor.conj()).real, axes, full_shape
    )

    result: list[_PendingBranch] = []
    for outcome, outcome_prob in enumerate(probabilities):
        if outcome_prob <= atol:
            continue
        new_state = _fork_state(state)
        new_tensor = new_state.target_tensor
        result_slice = linalg.slice_for_qubits_equal_to(
            axes, big_endian_qureg_value=outcome, qid_shape=full_shape
        )
        mask_array = np.ones(full_shape, dtype=bool)
        mask_array[result_slice] = False
        new_tensor[mask_array] = 0
        new_tensor /= np.sqrt(outcome_prob)
        flip_gate = common_gates.XPowGate(dimension=gate.dimension) ** (gate.dimension - outcome)
        protocols.act_on(flip_gate, new_state, list(qubits))
        result.append(
            _PendingBranch(
                probability=branch.probability * float(outcome_prob),
                state=new_state,
                path=(*branch.path, outcome),
            )
        )
    return result


def _mixture_branches(
    op: cirq.Operation, branch: _PendingBranch, atol: float
) -> list[_PendingBranch]:
    """Branches over the components of a mixture/random channel.

    If the mixture has a measurement key (a keyed random channel), the selected
    component index is recorded as a channel measurement, exactly like the
    state vector simulator does.
    """
    mixture = protocols.mixture(op, default=None)
    if mixture is None:  # pragma: no cover
        raise TypeError(f"Cannot exactly enumerate the mixture of {op!r}.")
    state = branch.state
    axes = state.get_axes(op.qubits)
    qid_shape = protocols.qid_shape(op)
    is_measurement = protocols.is_measurement(op)
    key = protocols.measurement_key_obj(op) if is_measurement else None

    result: list[_PendingBranch] = []
    for index, (component_probability, unitary) in enumerate(mixture):
        if component_probability <= atol:
            continue
        new_state = _fork_state(state)
        unitary_tensor = (
            np.asarray(unitary).astype(new_state.target_tensor.dtype).reshape(qid_shape * 2)
        )
        linalg.targeted_left_multiply(
            unitary_tensor,
            new_state.target_tensor,
            axes,
            out=new_state.available_buffer,
        )
        new_state._state._swap_target_tensor_for(new_state.available_buffer)
        if is_measurement:
            assert key is not None
            new_state._classical_data.record_channel_measurement(key, index)
        result.append(
            _PendingBranch(
                probability=branch.probability * float(component_probability),
                state=new_state,
                path=(*branch.path, index),
            )
        )
    return result


def _decompose_branches(
    raw: cirq.Operation, branch: _PendingBranch, atol: float
) -> list[_PendingBranch]:
    """Falls back to a one-shot decomposition, tracking any added ancillas."""
    if isinstance(raw, ops.GateOperation):
        decomposed = protocols.decompose_once_with_qubits(
            raw.gate, raw.qubits, flatten=False, default=None
        )
    else:
        decomposed = protocols.decompose_once(raw, flatten=False, default=None)
    if decomposed is None:
        raise TypeError(
            "Cannot exactly enumerate branches: the operation "
            f"{raw!r} is not a unitary, a mixture, a computational-basis "
            "measurement, a reset or a supported control-flow operation. "
            "General (non-mixture) Kraus channels cannot be represented as "
            "pure state branches."
        )

    # Decompositions may introduce ancilla qubits. This mirrors the state
    # vector simulator's decomposition strategy.
    all_ancilla: set[cirq.Qid] = set()
    branches = [branch]
    for inner_op in ops.flatten_to_ops(decomposed):
        next_branches: list[_PendingBranch] = []
        for inner_branch in branches:
            ancilla = tuple(q for q in inner_op.qubits if q not in inner_branch.state.qubit_map)
            if ancilla:
                grown = inner_branch.state.add_qubits(ancilla)
                if grown is NotImplemented:
                    raise TypeError(f"State cannot add ancilla qubits {ancilla} needed by {raw!r}.")
                inner_branch = _PendingBranch(
                    probability=inner_branch.probability,
                    state=grown,
                    path=inner_branch.path,
                )
                all_ancilla.update(ancilla)
            next_branches.extend(_apply_op(inner_op, inner_branch, atol))
        branches = next_branches
    if all_ancilla:
        for i, inner_branch in enumerate(branches):
            reduced = inner_branch.state.remove_qubits(tuple(all_ancilla))
            if reduced is NotImplemented:
                raise TypeError(
                    f"Ancilla qubits {all_ancilla} introduced while decomposing {raw!r} "
                    "could not be removed (they must end separable and back in |0>)."
                )
            branches[i] = _PendingBranch(
                probability=inner_branch.probability, state=reduced, path=inner_branch.path
            )
    return branches


def _classical_history_signature(
    classical_data: cirq.ClassicalDataStoreReader,
) -> tuple[tuple[str, int, tuple[Any, ...]], ...]:
    parts: list[tuple[str, int, tuple[Any, ...]]] = []
    for key in classical_data.keys():
        if key in classical_data.records:
            records: tuple[Any, ...] = tuple(classical_data.records[key])
            measurement_type = int(value.MeasurementType.MEASUREMENT)
        else:
            records = tuple(classical_data.channel_records[key])
            measurement_type = int(value.MeasurementType.CHANNEL)
        parts.append((str(key), measurement_type, records))
    return tuple(parts)


def _finalize_branches(branches: Sequence[_PendingBranch]) -> list[CircuitBranch]:
    """Merges equivalent converged branches and sorts them in stable order."""
    merged: list[_PendingBranch] = []
    vectors: list[np.ndarray] = []
    signatures: list[tuple[tuple[str, int, tuple[Any, ...]], ...]] = []
    for branch in sorted(branches, key=lambda b: b.path):
        final_vector = np.array(branch.state.target_tensor, copy=True).reshape(-1)
        signature = _classical_history_signature(branch.state.classical_data)
        for index, (existing_signature, existing_vector) in enumerate(zip(signatures, vectors)):
            if (
                existing_signature == signature
                and final_vector.shape == existing_vector.shape
                and np.allclose(final_vector, existing_vector, atol=1e-8)
            ):
                # Replace with a branch whose probability is the sum and whose
                # path stays the (already lexicographically smaller) one.
                merged[index] = _PendingBranch(
                    probability=merged[index].probability + branch.probability,
                    state=merged[index].state,
                    path=merged[index].path,
                )
                break
        else:
            merged.append(branch)
            vectors.append(final_vector)
            signatures.append(signature)

    result: list[CircuitBranch] = []
    for branch in sorted(merged, key=lambda b: b.path):
        final_vector = np.array(branch.state.target_tensor, copy=True).reshape(-1)
        result.append(
            CircuitBranch(
                probability=float(branch.probability),
                classical_data=branch.state.classical_data,
                final_state_vector=final_vector,
            )
        )
    return result


# Alias spelling out what the function does for users that find the short name
# ambiguous.
enumerate_dynamic_circuit_branches = enumerate_branches
