"""Stage 2: minimal controlled logical bit-flip noise experiment for BB84.

Uses the same circuit structure and QBER sampling rule as Stage 1. The noise is
applied as a bit flip in Alice's encoded basis: X for Z-basis states and Z for
X-basis states. This makes the injected error a logical bit error in either
basis. This is a simplified channel-noise model, not a complete physical
channel model or a finite-key security proof.
"""

import argparse
import csv
import random
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def run_bb84(n_qubits: int, seed: int, noise_probability: float,
             sample_fraction: float = 0.20,
             qber_threshold: float = 0.11) -> dict:
    if n_qubits < 256:
        raise ValueError("Use at least 256 transmitted qubits.")
    if not 0 <= noise_probability <= 1:
        raise ValueError("noise_probability must be between 0 and 1.")
    if not 0 < sample_fraction < 1:
        raise ValueError("sample_fraction must be between 0 and 1.")
    if not 0 <= qber_threshold <= 1:
        raise ValueError("qber_threshold must be between 0 and 1.")

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

        # Logical bit-flip in Alice's encoding basis.
        if apply_flip:
            if alice_basis == 0:  # Z basis
                qc.x(0)
            else:                 # X basis
                qc.z(0)

        # Bob rotates X-basis states back to the Z measurement basis.
        if bob_basis:
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)

    backend = AerSimulator()
    compiled = transpile(circuits, backend, optimization_level=0,
                          seed_transpiler=seed)
    sim_result = backend.run(compiled, shots=1, seed_simulator=seed).result()
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

    status = "ACCEPT_QBER" if qber <= qber_threshold else "REJECT_QBER"

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
        "remaining_raw_bits": len(alice_raw),
        "raw_keys_match": raw_keys_match,
        "sifting_efficiency_percent": round(100 * len(sifted_indices) / n_qubits, 2),
        "raw_256bit_chunks_upper_bound": len(alice_raw) // 256,
        "acceptance_status": status,
        "security_note": "Sampled QBER decision only; reconciliation and secure final-key estimation are not implemented."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.15])
    parser.add_argument("--sample-fraction", type=float, default=0.20)
    parser.add_argument("--qber-threshold", type=float, default=0.11)
    parser.add_argument("--output", type=Path,
                        default=Path("results/bb84_stage2_noise.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in args.noise_levels:
        start = time.perf_counter()
        row = run_bb84(args.qubits, args.seed, level,
                       args.sample_fraction, args.qber_threshold)
        row["execution_time_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)
        print(" | ".join(f"{key}={value}" for key, value in row.items()))

    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved Stage 2 results to: {args.output}")


if __name__ == "__main__":
    main()
