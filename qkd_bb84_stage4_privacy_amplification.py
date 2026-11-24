"""Stage 4: BB84 post-processing with Toeplitz privacy amplification.

Reproduces the Stage 3 BB84 data using the same defaults, performs the same
simplified multi-pass parity reconciliation, then applies Toeplitz universal
hashing to accepted, reconciled raw keys.

IMPORTANT RESEARCH LIMITATION:
The output length uses the manuscript's simplified asymptotic estimate:
    ell = n * (1 - H2(Q_sample)) - leak_EC
where leak_EC is the parity disclosure count from this implementation.
This is NOT a finite-key/composable security proof. The sampled QBER is not
a rigorous upper confidence bound on the phase-error rate. Therefore the
Toeplitz output is an experimental privacy-amplification demonstration and
must not be described as a proven secure final QKD key.
"""

import argparse
import csv
import hashlib
import math
import random
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def h2(q):
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def parity(bits, indices):
    return sum(bits[i] for i in indices) & 1


def cascade_reconcile(alice, bob, seed, passes, initial_block_size):
    """Same multi-pass parity reconciliation procedure as Stage 3."""
    bob_fixed = bob.copy()
    rng = random.Random(seed)
    disclosed = 0
    corrected = 0
    n = len(alice)

    for pass_index in range(passes):
        permutation = list(range(n))
        rng.shuffle(permutation)
        block_size = max(1, initial_block_size * (2 ** pass_index))

        for start in range(0, n, block_size):
            block = permutation[start:start + block_size]
            if not block:
                continue
            disclosed += 1
            if parity(alice, block) != parity(bob_fixed, block):
                lo, hi = 0, len(block)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    left = block[lo:mid]
                    disclosed += 1
                    if parity(alice, left) != parity(bob_fixed, left):
                        hi = mid
                    else:
                        lo = mid
                bob_fixed[block[lo]] ^= 1
                corrected += 1

    return bob_fixed, disclosed, corrected


def toeplitz_hash(input_bits, output_length, seed):
    """Universal-2 Toeplitz hashing over GF(2), returning output bits."""
    n = len(input_bits)
    m = output_length
    if m < 1 or m > n:
        raise ValueError("Toeplitz output length must be between 1 and input length.")

    rng = random.Random(seed)
    # An m x n Toeplitz matrix is defined by n + m - 1 random bits.
    toeplitz_seed = [rng.randrange(2) for _ in range(n + m - 1)]
    output = []
    for row in range(m):
        value = 0
        for col, bit in enumerate(input_bits):
            if bit:
                value ^= toeplitz_seed[row - col + n - 1]
        output.append(value)
    return output


def simulate_condition(n_qubits, seed, noise_probability, sample_fraction,
                       qber_threshold, f_ec, cascade_passes, pa_seed):
    rng = random.Random(seed)
    noise_rng = random.Random(seed + 900000)

    alice_bits = [rng.randrange(2) for _ in range(n_qubits)]
    alice_bases = [rng.randrange(2) for _ in range(n_qubits)]
    bob_bases = [rng.randrange(2) for _ in range(n_qubits)]
    noise_flags = [noise_rng.random() < noise_probability
                   for _ in range(n_qubits)]
    injected_flips = sum(noise_flags)

    circuits = []
    for bit, a_basis, b_basis, apply_flip in zip(
            alice_bits, alice_bases, bob_bases, noise_flags):
        qc = QuantumCircuit(1, 1)
        if bit:
            qc.x(0)
        if a_basis:
            qc.h(0)
        if apply_flip:
            qc.x(0) if a_basis == 0 else qc.z(0)
        if b_basis:
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)

    backend = AerSimulator()
    compiled = transpile(circuits, backend, optimization_level=0,
                         seed_transpiler=seed)
    result = backend.run(compiled, shots=1, seed_simulator=seed).result()
    bob_bits = [int(result.get_counts(i).most_frequent())
                for i in range(n_qubits)]

    sifted = [i for i in range(n_qubits) if alice_bases[i] == bob_bases[i]]
    sample_n = max(32, int(len(sifted) * sample_fraction))
    sample_set = set(rng.sample(sifted, sample_n))
    sample_errors = sum(alice_bits[i] != bob_bits[i] for i in sample_set)
    qber = sample_errors / sample_n

    remaining = [i for i in sifted if i not in sample_set]
    alice_raw = [alice_bits[i] for i in remaining]
    bob_raw = [bob_bits[i] for i in remaining]
    accepted = qber <= qber_threshold

    row = {
        "qubits_sent": n_qubits,
        "seed": seed,
        "noise_probability": noise_probability,
        "injected_logical_flips_all_qubits": injected_flips,
        "sifted_bits": len(sifted),
        "qber_sample_bits": sample_n,
        "qber_sample_errors": sample_errors,
        "sampled_qber": qber,
        "remaining_raw_bits": len(alice_raw),
        "raw_keys_match_before_reconciliation": alice_raw == bob_raw,
        "qber_decision": "ACCEPT" if accepted else "REJECT",
    }

    if not accepted:
        row.update({
            "reconciliation_performed": False,
            "reconciled_keys_match": False,
            "cascade_parity_bits_disclosed": 0,
            "fEC_model_leakage_estimate_bits_not_added": 0,
            "privacy_amplification_performed": False,
            "toeplitz_output_bits": 0,
            "toeplitz_output_sha256_fingerprint": "",
            "key_status": "ABORTED_QBER_REJECTED",
        })
        return row

    if qber == 0:
        bob_reconciled = bob_raw.copy()
        disclosed = 0
        corrected = 0
    else:
        initial_block_size = max(1, int(0.73 / qber))
        bob_reconciled, disclosed, corrected = cascade_reconcile(
            alice_raw, bob_raw, seed + 12345, cascade_passes,
            initial_block_size
        )

    match = alice_raw == bob_reconciled
    row.update({
        "reconciliation_performed": True,
        "cascade_passes": cascade_passes,
        "cascade_corrected_errors": corrected,
        "cascade_parity_bits_disclosed": disclosed,
        "reconciled_keys_match": match,
        "fEC": f_ec,
        "fEC_model_leakage_estimate_bits_not_added": math.ceil(
            f_ec * len(alice_raw) * h2(qber)
        ),
    })

    if not match:
        row.update({
            "privacy_amplification_performed": False,
            "toeplitz_output_bits": 0,
            "toeplitz_output_sha256_fingerprint": "",
            "key_status": "ABORTED_RECONCILIATION_FAILED",
        })
        return row

    # This deliberately implements the manuscript's simplified asymptotic
    # length estimate, using measured parity disclosure as leak_EC.
    # It is not a finite-key/composable security bound.
    n = len(alice_raw)
    leak_ec = disclosed
    estimated_length = math.floor(n * (1.0 - h2(qber)) - leak_ec)
    output_length = max(0, min(n, estimated_length))

    if output_length == 0:
        row.update({
            "privacy_amplification_performed": False,
            "toeplitz_output_bits": 0,
            "toeplitz_output_sha256_fingerprint": "",
            "simplified_asymptotic_estimated_output_bits": estimated_length,
            "key_status": "NO_POSITIVE_OUTPUT_LENGTH",
        })
        return row

    final_bits = toeplitz_hash(alice_raw, output_length, pa_seed)
    packed = bytearray((output_length + 7) // 8)
    for i, bit in enumerate(final_bits):
        if bit:
            packed[i // 8] |= 1 << (7 - (i % 8))
    fingerprint = hashlib.sha256(bytes(packed)).hexdigest()

    row.update({
        "privacy_amplification_performed": True,
        "simplified_asymptotic_estimated_output_bits": estimated_length,
        "toeplitz_output_bits": output_length,
        "toeplitz_seed_bits_required": n + output_length - 1,
        "toeplitz_output_sha256_fingerprint": fingerprint,
        "key_status": "PRIVACY_AMPLIFICATION_DEMONSTRATED_NOT_FINITE_KEY_PROVEN",
    })
    # Kept only in memory for optional export; removed before writing the CSV.
    row["_toeplitz_output_bits_internal"] = final_bits
    return row


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
    parser.add_argument("--pa-seed", type=int, default=424242)
    parser.add_argument("--output", type=Path,
                        default=Path("results/bb84_stage4_privacy_amplification.csv"))
    parser.add_argument("--master-secret-output", type=Path,
                        default=Path("qkd_master_secret.hex"))
    parser.add_argument("--master-secret-bits", type=int, default=256)
    args = parser.parse_args()

    if args.master_secret_bits <= 0:
        parser.error("--master-secret-bits must be a positive integer")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    exported_master_secret = False
    for noise in args.noise_levels:
        start = time.perf_counter()
        row = simulate_condition(
            args.qubits, args.seed, noise, args.sample_fraction,
            args.qber_threshold, args.f_ec, args.cascade_passes, args.pa_seed
        )
        row["execution_time_seconds"] = round(time.perf_counter() - start, 3)

        # Export the real Toeplitz output from the ideal (0% noise) condition.
        # The secret itself is never printed or written to the results CSV.
        internal_bits = row.pop("_toeplitz_output_bits_internal", None)
        if noise == 0.0 and row.get("privacy_amplification_performed") and internal_bits is not None:
            if len(internal_bits) < args.master_secret_bits:
                raise ValueError(
                    f"Toeplitz output has {len(internal_bits)} bits, fewer than "
                    f"the requested {args.master_secret_bits}-bit master secret."
                )
            selected_bits = internal_bits[:args.master_secret_bits]
            packed_secret = bytearray((args.master_secret_bits + 7) // 8)
            for i, bit in enumerate(selected_bits):
                if bit:
                    packed_secret[i // 8] |= 1 << (7 - (i % 8))

            # This hybrid script expects 256 bits = 32 bytes = 64 hex characters.
            if args.master_secret_bits % 8 != 0:
                raise ValueError(
                    "For the hybrid script, --master-secret-bits must be "
                    "a multiple of 8 (default: 256)."
                )

            args.master_secret_output.parent.mkdir(parents=True, exist_ok=True)
            args.master_secret_output.write_text(
                bytes(packed_secret).hex() + "\\n", encoding="ascii"
            )
            exported_master_secret = True
            print(
                f"Exported {args.master_secret_bits}-bit QKD master secret to: "
                f"{args.master_secret_output} (secret value not displayed)"
            )

        rows.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()))

    if not exported_master_secret:
        raise RuntimeError(
            "No QKD master secret was exported. Ensure the 0% noise condition "
            "is included and privacy amplification succeeds."
        )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved Stage 4 results to: {args.output}")
    print("WARNING: Toeplitz extraction here demonstrates privacy amplification "
          "under the manuscript's simplified asymptotic estimate only; "
          "it is not a finite-key/composable security proof.")


if __name__ == "__main__":
    main()
