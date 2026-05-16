#!/usr/bin/env python3
"""Run synthetic acoustic/OFDM experiments and generate figures.

Usage from the project root:
    python3 scripts/run_experiments.py

Outputs:
    figures/*.pdf and figures/*.png
    results/ber_summary.csv
    results/decoded_message.txt
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_ofdm.ofdm_utils import (  # noqa: E402
    add_awgn,
    apply_channel,
    bit_error_rate,
    bits_to_text,
    bpsk_demod,
    bpsk_mod,
    channel_frequency_response,
    default_active_subcarriers,
    multipath_channel,
    ofdm_demodulate,
    ofdm_modulate,
    pam_matched_filter_detect,
    pam_transmit,
    rng_from_seed,
    text_to_bits,
)

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def savefig(name: str) -> None:
    """Save current matplotlib figure as PDF and PNG."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.pdf")
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close()


def plot_channel(h: np.ndarray, n_fft: int, active: np.ndarray) -> None:
    n = np.arange(len(h))
    plt.figure(figsize=(5.0, 3.2))
    markerline, stemlines, baseline = plt.stem(n, np.abs(h))
    plt.setp(baseline, linewidth=0.8)
    plt.xlabel("Delay sample $n$")
    plt.ylabel(r"$|h[n]|$")
    plt.title("Synthetic multipath impulse response")
    plt.grid(True, alpha=0.3)
    savefig("channel_impulse_response")

    H = np.fft.fftshift(np.fft.fft(h, 2048))
    f = np.linspace(-0.5, 0.5, len(H), endpoint=False)
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(f, 20 * np.log10(np.abs(H) + 1e-12))
    plt.xlabel("Normalized frequency")
    plt.ylabel(r"$20\log_{10}|H(e^{j\omega})|$ (dB)")
    plt.title("Frequency-selective channel response")
    plt.grid(True, alpha=0.3)
    savefig("channel_frequency_response")

    H_active = channel_frequency_response(h, n_fft, active)
    plt.figure(figsize=(5.0, 3.2))
    plt.stem(active, np.abs(H_active))
    plt.xlabel("OFDM subcarrier index $k$")
    plt.ylabel(r"$|H[k]|$")
    plt.title("Channel samples on active subcarriers")
    plt.grid(True, alpha=0.3)
    savefig("active_subcarrier_channel")


def plot_ofdm_spectrum(rng: np.random.Generator, n_fft: int, n_cp: int, active: np.ndarray) -> None:
    bits = rng.integers(0, 2, size=52 * 80, dtype=np.uint8)
    x, _, _ = ofdm_modulate(bpsk_mod(bits), n_fft=n_fft, n_cp=n_cp, active=active)
    n_win = min(4096, len(x))
    X = np.fft.fftshift(np.fft.fft(x[:n_win] * np.hanning(n_win), 8192))
    f = np.linspace(-0.5, 0.5, len(X), endpoint=False)
    psd = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-12)
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(f, psd)
    plt.xlabel("Normalized frequency")
    plt.ylabel("Magnitude (dB, normalized)")
    plt.title("OFDM baseband spectrum")
    plt.ylim([-80, 5])
    plt.grid(True, alpha=0.3)
    savefig("ofdm_spectrum")


def plot_cyclic_prefix_property(rng: np.random.Generator, h: np.ndarray, n_fft: int, active: np.ndarray) -> None:
    bits = rng.integers(0, 2, size=len(active), dtype=np.uint8)
    s = bpsk_mod(bits)
    H_active = channel_frequency_response(h, n_fft, active)
    cp_values = np.arange(0, 25)
    rel_errors = []
    for n_cp in cp_values:
        x, n_syms, _ = ofdm_modulate(s, n_fft=n_fft, n_cp=int(n_cp), active=active)
        y = apply_channel(x, h, keep_len=True)
        S_raw = ofdm_demodulate(y, n_syms, n_fft=n_fft, n_cp=int(n_cp), active=active, h_freq=None)
        ideal = H_active * s
        err = np.linalg.norm(S_raw[: len(s)] - ideal) / np.linalg.norm(ideal)
        rel_errors.append(err)

    plt.figure(figsize=(5.0, 3.2))
    plt.semilogy(cp_values, rel_errors, marker="o")
    plt.axvline(len(h) - 1, linestyle="--")
    plt.xlabel("Cyclic prefix length $N_{cp}$")
    plt.ylabel("Relative DFT-domain model error")
    plt.title("CP length needed for circular convolution model")
    plt.grid(True, which="both", alpha=0.3)
    savefig("cyclic_prefix_error")


def simulate_ber_curves(
    rng: np.random.Generator,
    h: np.ndarray,
    n_fft: int,
    n_cp: int,
    active: np.ndarray,
) -> list[dict[str, float]]:
    snrs = np.arange(-2, 23, 2)
    n_bits = len(active) * 350
    H_active = channel_frequency_response(h, n_fft, active)

    results = []
    for snr_db in snrs:
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        syms = bpsk_mod(bits)

        # OFDM with CP and perfect one-tap equalization.
        x_cp, n_syms_cp, n_payload = ofdm_modulate(syms, n_fft=n_fft, n_cp=n_cp, active=active)
        y_cp = apply_channel(x_cp, h, keep_len=True)
        y_cp = add_awgn(y_cp, float(snr_db), rng)
        shat = ofdm_demodulate(y_cp, n_syms_cp, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=H_active)
        ber_cp_eq = bit_error_rate(bits, bpsk_demod(shat[:n_payload]))

        # OFDM with CP but no channel equalization.
        shat_noeq = ofdm_demodulate(y_cp, n_syms_cp, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=None)
        ber_cp_noeq = bit_error_rate(bits, bpsk_demod(shat_noeq[:n_payload]))

        # OFDM without CP, but still using the nominal H[k]. This isolates the
        # loss caused by the missing circular-convolution guard interval.
        x_nocp, n_syms_nocp, n_payload_nocp = ofdm_modulate(syms, n_fft=n_fft, n_cp=0, active=active)
        y_nocp = apply_channel(x_nocp, h, keep_len=True)
        y_nocp = add_awgn(y_nocp, float(snr_db), rng)
        shat_nocp = ofdm_demodulate(y_nocp, n_syms_nocp, n_fft=n_fft, n_cp=0, active=active, h_freq=H_active)
        ber_nocp_eq = bit_error_rate(bits, bpsk_demod(shat_nocp[:n_payload_nocp]))

        # CP-OFDM with a noisy one-symbol least-squares channel estimate.
        train = np.ones(len(active), dtype=np.complex128)
        x_train, _, _ = ofdm_modulate(train, n_fft=n_fft, n_cp=n_cp, active=active)
        x_frame = np.concatenate([x_train, x_cp])
        y_frame = apply_channel(x_frame, h, keep_len=True)
        y_frame = add_awgn(y_frame, float(snr_db), rng)
        raw = ofdm_demodulate(y_frame, n_syms_cp + 1, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=None)
        raw_blocks = raw.reshape(n_syms_cp + 1, len(active))
        H_est = raw_blocks[0] / train
        payload_eq = (raw_blocks[1:] / np.where(np.abs(H_est) < 1e-12, 1e-12, H_est)).reshape(-1)
        ber_cp_est = bit_error_rate(bits, bpsk_demod(payload_eq[:n_payload]))

        # Single-carrier rectangular PAM with a matched filter, in AWGN and in
        # the same multipath environment without a decision-feedback/equalizer.
        x_pam, pulse = pam_transmit(syms, sps=8)
        y_pam_awgn = add_awgn(x_pam, float(snr_db), rng)
        bits_pam_awgn = pam_matched_filter_detect(y_pam_awgn, pulse, len(bits), sps=8)
        ber_pam_awgn = bit_error_rate(bits, bits_pam_awgn)

        # A deliberately echo-rich real channel for the single-carrier baseline.
        # The delays are comparable to or larger than the rectangular pulse width,
        # so a receiver with only a matched filter has residual ISI.
        h_pam = np.zeros(33, dtype=float)
        h_pam[0] = 1.0
        # Echoes spaced at roughly one and two symbol periods create a clear
        # high-SNR ISI floor for a receiver that only uses a matched filter.
        h_pam[8] = 0.90
        h_pam[16] = -0.75
        h_pam[24] = 0.35
        h_pam = h_pam / np.sqrt(np.sum(h_pam**2))
        y_pam_mp = apply_channel(x_pam, h_pam, keep_len=True)
        y_pam_mp = add_awgn(y_pam_mp, float(snr_db), rng)
        bits_pam_mp = pam_matched_filter_detect(y_pam_mp, pulse, len(bits), sps=8)
        ber_pam_mp = bit_error_rate(bits, bits_pam_mp)

        results.append(
            {
                "snr_db": float(snr_db),
                "ofdm_cp_true_eq": ber_cp_eq,
                "ofdm_cp_ls_eq": ber_cp_est,
                "ofdm_no_cp_true_eq": ber_nocp_eq,
                "ofdm_cp_no_eq": ber_cp_noeq,
                "pam_awgn_matched_filter": ber_pam_awgn,
                "pam_multipath_matched_filter": ber_pam_mp,
            }
        )
    return results


def plot_ber(results: list[dict[str, float]], n_bits_floor: int) -> None:
    snr = np.array([row["snr_db"] for row in results])
    keys = [
        "ofdm_cp_true_eq",
        "ofdm_cp_ls_eq",
        "ofdm_no_cp_true_eq",
        "ofdm_cp_no_eq",
        "pam_awgn_matched_filter",
        "pam_multipath_matched_filter",
    ]
    labels = {
        "ofdm_cp_true_eq": "OFDM, CP + true one-tap EQ",
        "ofdm_cp_ls_eq": "OFDM, CP + training EQ",
        "ofdm_no_cp_true_eq": "OFDM, no CP + nominal EQ",
        "ofdm_cp_no_eq": "OFDM, CP but no EQ",
        "pam_awgn_matched_filter": "PAM matched filter, AWGN",
        "pam_multipath_matched_filter": "PAM matched filter, multipath",
    }
    floor = 0.5 / n_bits_floor
    plt.figure(figsize=(6.0, 4.0))
    for key in keys:
        ber = np.array([row[key] for row in results])
        plt.semilogy(snr, np.maximum(ber, floor), marker="o", label=labels[key])
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit error rate")
    plt.title("BER comparison under noise and multipath")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=7)
    savefig("ber_curves")


def write_ber_csv(results: list[dict[str, float]]) -> None:
    path = RES_DIR / "ber_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def summarize_ber_trials(trial_results: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    """Average BER curves over independent Monte Carlo trials."""
    if not trial_results:
        raise ValueError("Need at least one trial result.")
    keys = [k for k in trial_results[0][0].keys() if k != "snr_db"]
    n_trials = len(trial_results)
    n_snrs = len(trial_results[0])
    summary: list[dict[str, float]] = []
    for i in range(n_snrs):
        row: dict[str, float] = {"snr_db": float(trial_results[0][i]["snr_db"])}
        for key in keys:
            vals = np.array([trial_results[t][i][key] for t in range(n_trials)], dtype=float)
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0
            row[f"{key}_se"] = float(row[f"{key}_std"] / np.sqrt(n_trials)) if n_trials > 1 else 0.0
            row[f"{key}_ci95"] = float(1.96 * row[f"{key}_se"])
        summary.append(row)
    return summary


def write_ber_mc_csv(summary: list[dict[str, float]]) -> None:
    """Write Monte Carlo BER summary with mean, standard error, and 95% CI."""
    path = RES_DIR / "ber_summary_mc.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_ber_mc(summary: list[dict[str, float]], n_bits_floor: int) -> None:
    """Plot mean BER curves from the Monte Carlo summary."""
    snr = np.array([row["snr_db"] for row in summary])
    keys = [
        "ofdm_cp_true_eq",
        "ofdm_cp_ls_eq",
        "ofdm_no_cp_true_eq",
        "ofdm_cp_no_eq",
        "pam_awgn_matched_filter",
        "pam_multipath_matched_filter",
    ]
    labels = {
        "ofdm_cp_true_eq": "OFDM, CP + true one-tap EQ",
        "ofdm_cp_ls_eq": "OFDM, CP + training EQ",
        "ofdm_no_cp_true_eq": "OFDM, no CP + nominal EQ",
        "ofdm_cp_no_eq": "OFDM, CP but no EQ",
        "pam_awgn_matched_filter": "PAM matched filter, AWGN",
        "pam_multipath_matched_filter": "PAM matched filter, multipath",
    }
    floor = 0.5 / n_bits_floor
    plt.figure(figsize=(6.0, 4.0))
    for key in keys:
        ber = np.array([row[f"{key}_mean"] for row in summary])
        plt.semilogy(snr, np.maximum(ber, floor), marker="o", label=labels[key])
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit error rate")
    plt.title("Mean BER over independent packets")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=7)
    savefig("ber_curves")


def simulate_cp_length_sweep(
    h: np.ndarray,
    n_fft: int,
    active: np.ndarray,
    snr_db: float = 18.0,
    n_trials: int = 20,
) -> list[dict[str, float]]:
    """Estimate BER as a function of cyclic-prefix length at one SNR."""
    cp_values = np.array([0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 20, 24], dtype=int)
    n_bits = len(active) * 300
    H_active = channel_frequency_response(h, n_fft, active)
    rows: list[dict[str, float]] = []
    for n_cp in cp_values:
        vals = []
        for trial in range(n_trials):
            rng = rng_from_seed(9100 + 97 * trial + int(n_cp))
            bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
            syms = bpsk_mod(bits)
            x, n_syms, n_payload = ofdm_modulate(syms, n_fft=n_fft, n_cp=int(n_cp), active=active)
            y = apply_channel(x, h, keep_len=True)
            y = add_awgn(y, snr_db, rng)
            shat = ofdm_demodulate(y, n_syms, n_fft=n_fft, n_cp=int(n_cp), active=active, h_freq=H_active)
            vals.append(bit_error_rate(bits, bpsk_demod(shat[:n_payload])))
        vals = np.array(vals, dtype=float)
        rows.append(
            {
                "n_cp": int(n_cp),
                "snr_db": float(snr_db),
                "ber_mean": float(np.mean(vals)),
                "ber_std": float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0,
                "ber_se": float(np.std(vals, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "ber_ci95": float(1.96 * np.std(vals, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
                "r_eff": float(len(active) / (n_fft + n_cp)),
            }
        )
    return rows


def write_cp_sweep_csv(rows: list[dict[str, float]]) -> None:
    path = RES_DIR / "cp_length_sweep.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_cp_length_sweep(rows: list[dict[str, float]], channel_len: int) -> None:
    n_cp = np.array([row["n_cp"] for row in rows], dtype=int)
    ber = np.array([row["ber_mean"] for row in rows], dtype=float)
    ci = np.array([row["ber_ci95"] for row in rows], dtype=float)
    floor = 0.5 / (len(default_active_subcarriers(64)) * 300)
    plt.figure(figsize=(5.4, 3.4))
    plt.semilogy(n_cp, np.maximum(ber, floor), marker="o")
    lo = np.maximum(ber - ci, floor)
    hi = np.maximum(ber + ci, floor)
    plt.fill_between(n_cp, lo, hi, alpha=0.18)
    plt.axvline(channel_len - 1, linestyle="--", label=r"$L-1$")
    plt.xlabel("Cyclic prefix length $N_{cp}$")
    plt.ylabel("BER at 18 dB")
    plt.title("BER penalty when the cyclic prefix is too short")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    savefig("cp_length_ber_sweep")


def plot_constellations(rng: np.random.Generator, h: np.ndarray, n_fft: int, n_cp: int, active: np.ndarray) -> None:
    bits = rng.integers(0, 2, size=len(active) * 120, dtype=np.uint8)
    syms = bpsk_mod(bits)
    x, n_syms, n_payload = ofdm_modulate(syms, n_fft=n_fft, n_cp=n_cp, active=active)
    y = apply_channel(x, h, keep_len=True)
    y = add_awgn(y, 12.0, rng)
    raw = ofdm_demodulate(y, n_syms, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=None)[:n_payload]
    H_active = channel_frequency_response(h, n_fft, active)
    eq = ofdm_demodulate(y, n_syms, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=H_active)[:n_payload]

    n_show = min(1200, len(raw))
    plt.figure(figsize=(4.2, 4.0))
    plt.scatter(np.real(raw[:n_show]), np.imag(raw[:n_show]), s=8, alpha=0.5)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title("Received subcarriers before equalization")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    savefig("constellation_before_equalization")

    plt.figure(figsize=(4.2, 4.0))
    plt.scatter(np.real(eq[:n_show]), np.imag(eq[:n_show]), s=8, alpha=0.5)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title("Received subcarriers after one-tap equalization")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    savefig("constellation_after_equalization")


def decode_message_demo(rng: np.random.Generator, h: np.ndarray, n_fft: int, n_cp: int, active: np.ndarray) -> None:
    message = "ES156 OFDM DEMO: FFT EQUALIZATION CAN UNDO A MULTIPATH CHANNEL."
    bits = text_to_bits(message)
    syms = bpsk_mod(bits)
    x, n_syms, n_payload = ofdm_modulate(syms, n_fft=n_fft, n_cp=n_cp, active=active)
    y = apply_channel(x, h, keep_len=True)
    y = add_awgn(y, 28.0, rng)
    H_active = channel_frequency_response(h, n_fft, active)
    shat = ofdm_demodulate(y, n_syms, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=H_active)
    bits_hat = bpsk_demod(shat[:n_payload])[: len(bits)]
    decoded = bits_to_text(bits_hat)
    ber = bit_error_rate(bits, bits_hat)
    with open(RES_DIR / "decoded_message.txt", "w") as f:
        f.write("Original message:\n")
        f.write(message + "\n\n")
        f.write("Decoded message after synthetic multipath channel at 28 dB SNR:\n")
        f.write(decoded + "\n\n")
        f.write(f"Bit error rate: {ber:.6g}\n")


def main() -> None:
    os.chdir(ROOT)
    rng = rng_from_seed(156)
    n_fft = 64
    n_cp = 16
    active = default_active_subcarriers(n_fft)
    h = multipath_channel()

    plot_channel(h, n_fft, active)
    plot_ofdm_spectrum(rng, n_fft, n_cp, active)
    plot_cyclic_prefix_property(rng, h, n_fft, active)
    plot_constellations(rng, h, n_fft, n_cp, active)

    # Keep one single-run CSV for transparency, but plot and report the
    # Monte Carlo mean to make the quantitative claims less seed-dependent.
    results = simulate_ber_curves(rng, h, n_fft, n_cp, active)
    write_ber_csv(results)

    n_mc_trials = 20
    trial_results = []
    for trial in range(n_mc_trials):
        trial_rng = rng_from_seed(156 + 1000 * trial)
        trial_results.append(simulate_ber_curves(trial_rng, h, n_fft, n_cp, active))
    summary = summarize_ber_trials(trial_results)
    write_ber_mc_csv(summary)
    plot_ber_mc(summary, n_bits_floor=len(active) * 350 * n_mc_trials)

    cp_rows = simulate_cp_length_sweep(h, n_fft, active, snr_db=18.0, n_trials=n_mc_trials)
    write_cp_sweep_csv(cp_rows)
    plot_cp_length_sweep(cp_rows, channel_len=len(h))

    decode_message_demo(rng, h, n_fft, n_cp, active)

    print("Wrote figures to", FIG_DIR)
    print("Wrote results to", RES_DIR)
    print(f"Monte Carlo BER trials: {n_mc_trials}")


if __name__ == "__main__":
    main()
