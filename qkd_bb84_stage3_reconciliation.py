"""Stage 3: minimal BB84 information-reconciliation extension.

This script preserves the Stage 2 BB84 generation, sampling, and channel model.
For QBER-accepted runs, it applies a compact multi-pass Cascade-style parity
reconciliation procedure to the unsampled sifted strings.

Important:
- This is a simulation of classical post-processing, not a finite-key proof.
- `cascade_parity_bits_disclosed` counts parity bits revealed by this
  implementation (a leakage accounting measure).
- `fEC * n * H2(QBER)` is reported separately as a model-based estimate.
  These two values must NOT be added together.
- The QBER used to choose the initial Cascade block size is the sampled QBER.
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def binary_entropy(q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def run_bb84(n_qubits: int, seed: int, noise_probability: float,
             sample_fraction: float = 0.20,
             qber_threshold: float = 0.11,
             f_ec: float = 1.16,
             cascade_passes: int = 6) -> dict:
    if n_qubits < 256:
        raise ValueError("Use at least 256 transmitted qubits.")
    if not 0 <= noise_probability <= 1:
        raise ValueError("noise_probability must be between 0 and 1.")
    if not 0 < sample_fraction < 1:
        raise ValueError("sample_fraction must be between 0 and 1.")
    if not 0 <= qber_threshold <= 1:
        raise ValueError("qber_threshold must be between 0 and 1.")
    if f_ec < 1.0:
        raise ValueError("f_ec should be >= 1.")
    if cascade_passes < 1:
        raise ValueError("cascade_passes must be >= 1.")

    rng = random.Random(seed)
    noise_rng = random.Random(seed + 900000)
    alice_bits = [rng.randrange(2) for _ in range(n_qubits)]
    alice_bases = [rng.randrange(2) for _ in range(n_qubits)]
    bob_bases = [rng.randrange(2) for _ in range(n_qubits)]

    noise_flags = [noise_rng.random() < noise_probability
                   for _ in range(n_qubits)]
    injected_flips = sum(noise_flags)

    circuits = []
    for bit, alice_basis, bob_basis, apply_flip in zip(
            alice_bits, alice_bases, bob_bases, noise_flags):
        qc = QuantumCircuit(1, 1)
        if bit:
            qc.x(0)
        if alice_basis:
            qc.h(0)
        if apply_flip:
            if alice_basis == 0:
                qc.x(0)
            else:
                qc.z(0)
        if bob_basis:
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)

    backend = AerSimulator()
    compiled = transpile(circuits, backend, optimization_level=0,
                         seed_transpiler=seed)
    sim_result = backend.run(compiled, shots=1,
                             seed_simulator=seed).result()
    bob_bits = [int(sim_result.get_counts(i).most_frequent())
                for i in range(n_qubits)]

    sifted_indices = [i for i in range(n_qubits)
                      if alice_bases[i] == bob_bases[i]]
    sample_n = max(32, int(len(sifted_indices) * sample_fraction))
    if sample_n >= len(sifted_indices):
        raise RuntimeError("Not enough sifted bits for the requested QBER sample.")

    sample_indices = set(rng.sample(sifted_indices, sample_n))
    qber_errors = sum(alice_bits[i] != bob_bits[i] for i in sample_indices)
    qber = qber_errors / sample_n

    remaining_indices = [i for i in sifted_indices if i not in sample_indices]
    alice_raw = [alice_bits[i] for i in remaining_indices]
    bob_raw = [bob_bits[i] for i in remaining_indices]
    raw_keys_match = alice_raw == bob_raw

    accepted = qber <= qber_threshold
    reconciliation_performed = False
    corrected_errors = 0
    parity_bits_disclosed = 0
    reconciled_keys_match = False
    cascade_initial_block_size = 0
    reconciliation_status = "NOT_RUN_QBER_REJECTED"

    if accepted:
        reconciliation_performed = True
        if qber == 0:
            bob_reconciled = bob_raw.copy()
            reconciliation_status = "NO_ERRORS_OBSERVED"
        else:
            # Standard Cascade heuristic: initial block length is inversely
            # proportional to the estimated QBER. Later passes double it.
            cascade_initial_block_size = max(1, int(0.73 / qber))
            bob_reconciled, parity_bits_disclosed, corrected_errors = (
                cascade_reconcile(
                    alice_raw, bob_raw, qber, seed + 12345,
                    cascade_passes, cascade_initial_block_size
                )
            )
            reconciliation_status = "COMPLETED" if (
                alice_raw == bob_reconciled
            ) else "RESIDUAL_MISMATCH"
        reconciled_keys_match = alice_raw == bob_reconciled

    # Model-based leakage estimate, reported separately from actual Cascade
    # parity disclosures. Do not sum these two values.
    n_raw = len(alice_raw)
    f_ec_leakage_estimate = math.ceil(
        f_ec * n_raw * binary_entropy(qber)
    ) if accepted else 0

    return {
        "qubits_sent": n_qubits,
        "seed": seed,
        "channel_model": "logical_bit_flip",
        "noise_probability": noise_probability,
        "injected_logical_flips_all_qubits": injected_flips,
        "sample_fraction": sample_fraction,
        "qber_threshold": qber_threshold,
        "sifted_bits": len(sifted_indices),
        "qber_sample_bits": sample_n,
        "qber_sample_errors": qber_errors,
        "qber": round(qber, 6),
        "remaining_raw_bits": n_raw,
        "raw_keys_match_before_reconciliation": raw_keys_match,
        "acceptance_status": "ACCEPT_QBER" if accepted else "REJECT_QBER",
        "reconciliation_performed": reconciliation_performed,
        "cascade_passes": cascade_passes if accepted else 0,
        "cascade_initial_block_size": cascade_initial_block_size,
        "cascade_corrected_errors": corrected_errors,
        "cascade_parity_bits_disclosed": parity_bits_disclosed,
        "fEC": f_ec,
        "fEC_leakage_estimate_bits_separate": f_ec_leakage_estimate,
        "reconciliation_status": reconciliation_status,
        "reconciled_keys_match": reconciled_keys_match,
        "sifting_efficiency_percent": round(
            100 * len(sifted_indices) / n_qubits, 2
        ),
        "security_note": (
            "Reconciliation simulation only; no finite-key security bound "
            "or secure final-key length is claimed."
        ),
    }


def cascade_reconcile(alice: list[int], bob: list[int], qber: float,
                     seed: int, passes: int, initial_block_size: int):
    """Multi-pass Cascade-style parity reconciliation.

    Each block parity is compared. For a mismatching block, binary search
    locates one discrepant bit, which is flipped in Bob's string. Subsequent
    randomly permuted passes expose additional errors.
    """
    if len(alice) != len(bob):
        raise ValueError("Alice and Bob strings must have equal length.")

    bob_fixed = bob.copy()
    rng = random.Random(seed)
    disclosed = 0
    corrected = 0
    n = len(alice)

    def parity(bits, indices):
        return sum(bits[i] for i in indices) & 1

    for pass_index in range(passes):
        permutation = list(range(n))
        rng.shuffle(permutation)
        block_size = max(1, initial_block_size * (2 ** pass_index))

        for start in range(0, n, block_size):
            block = permutation[start:start + block_size]
            if not block:
                continue

            alice_parity = parity(alice, block)
            bob_parity = parity(bob_fixed, block)
            disclosed += 1

            if alice_parity != bob_parity:
                lo, hi = 0, len(block)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    left_half = block[lo:mid]
                    disclosed += 1
                    if parity(alice, left_half) != parity(bob_fixed, left_half):
                        hi = mid
                    else:
                        lo = mid

                error_index = block[lo]
                bob_fixed[error_index] ^= 1
                corrected += 1

    return bob_fixed, disclosed, corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.15])
    parser.add_argument("--sample-fraction", type=float, default=0.20)
    parser.add_argument("--qber-threshold", type=float, default=0.11)
    parser.add_argument("--f-ec", type=float, default=1.16)
    parser.add_argument("--cascade-passes", type=int, default=6)
    parser.add_argument("--output", type=Path,
                        default=Path("results/bb84_stage3_reconciliation.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in args.noise_levels:
        start = time.perf_counter()
        row = run_bb84(
            args.qubits, args.seed, level, args.sample_fraction,
            args.qber_threshold, args.f_ec, args.cascade_passes
        )
        row["execution_time_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)
        print(" | ".join(f"{key}={value}" for key, value in row.items()))

    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved Stage 3 results to: {args.output}")


if __name__ == "__main__":
    main()
