#!/usr/bin/env python3
"""Optional simulated audio OFDM packet demo.

Run from project root:
    python3 scripts/run_audio_loopback_demo.py

This writes audio/tx_packet.wav and audio/rx_loopback_simulated.wav.  It also
attempts to decode the simulated received packet and generates two figures.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_ofdm.audio_utils import (  # noqa: E402
    AudioConfig,
    decode_packet,
    encode_packet,
    synthetic_audio_channel,
    write_wav,
)

FIG_DIR = ROOT / "figures"
AUD_DIR = ROOT / "audio"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
AUD_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.pdf")
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close()


def main() -> None:
    cfg = AudioConfig()
    rng = np.random.default_rng(156)
    message = "ES156 audio OFDM: the FFT diagonalizes a cyclic convolution."
    tx, meta = encode_packet(message, cfg)
    rx, h = synthetic_audio_channel(tx, rng=rng, snr_db=26.0)
    decoded, diag = decode_packet(rx, cfg, max_data_blocks=meta["data_blocks"])

    write_wav(AUD_DIR / "tx_packet.wav", tx, cfg.fs)
    write_wav(AUD_DIR / "rx_loopback_simulated.wav", rx, cfg.fs)

    # Plot a short window around the preamble onset.
    n0 = int(cfg.silence_s * cfg.fs)
    n1 = n0 + int(0.05 * cfg.fs)
    t_ms = 1000 * np.arange(n1 - n0) / cfg.fs
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(t_ms, tx[n0:n1], label="transmitted")
    plt.plot(t_ms, rx[n0:n1], label="received", alpha=0.8)
    plt.xlabel("time after preamble start (ms)")
    plt.ylabel("amplitude")
    plt.title("Audio OFDM packet through simulated loopback")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, alpha=0.3)
    savefig("audio_loopback_waveform")

    H = np.fft.rfft(h, 4096)
    freqs = np.fft.rfftfreq(4096, d=1/cfg.fs)
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(freqs / 1000, 20 * np.log10(np.abs(H) + 1e-12))
    plt.xlabel("frequency (kHz)")
    plt.ylabel("magnitude (dB)")
    plt.title("Simulated speaker-room-microphone response")
    plt.grid(True, alpha=0.3)
    savefig("audio_loopback_channel_response")

    with open(RES_DIR / "audio_loopback_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Original: {message}\n")
        f.write(f"Decoded:  {decoded}\n")
        f.write(f"Packet metadata: {meta}\n")
        f.write(f"Decoder diagnostics: {diag}\n")

    print("Original:", message)
    print("Decoded: ", decoded)
    print("Metadata:", meta)
    print("Diagnostics:", diag)


if __name__ == "__main__":
    main()
