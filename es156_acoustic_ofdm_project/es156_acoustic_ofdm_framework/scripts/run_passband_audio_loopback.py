#!/usr/bin/env python3
"""Optional passband audio loopback demo.

This script turns the complex-baseband OFDM packet into a real-valued audio
waveform, passes it through a simulated speaker/room/microphone channel, mixes it
back down, estimates the subcarrier channel from one training symbol, and decodes
a text message. It is not a full real-time modem because it assumes the packet
start is known. That limitation is useful to state in the write-up.

Usage:
    python3 scripts/run_passband_audio_loopback.py

Outputs:
    audio/tx_packet.wav
    audio/rx_loopback.wav
    figures/passband_audio_spectrum.pdf and .png
    results/passband_audio_demo.txt
"""
from __future__ import annotations

import sys
import wave
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
    bits_to_text,
    bpsk_demod,
    bpsk_mod,
    bit_error_rate,
    ofdm_demodulate,
    ofdm_modulate,
    rng_from_seed,
    text_to_bits,
)

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
AUDIO_DIR = ROOT / "audio"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


def write_wav(path: Path, samples: np.ndarray, fs: int) -> None:
    x = np.asarray(samples, dtype=float)
    peak = np.max(np.abs(x)) + 1e-12
    x16 = np.int16(np.clip(0.90 * x / peak, -1.0, 1.0) * 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(x16.tobytes())


def fft_lowpass(x: np.ndarray, fs: int, cutoff_hz: float) -> np.ndarray:
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1 / fs)
    X[np.abs(freqs) > cutoff_hz] = 0.0
    return np.fft.ifft(X)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.pdf")
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close()


def main() -> None:
    rng = rng_from_seed(156)
    fs = 44100
    fc = 8000.0
    n_fft = 256
    n_cp = 64
    active = np.r_[np.arange(-12, 0), np.arange(1, 13)]

    message = "ES156 PASSBAND AUDIO OFDM LOOPBACK"
    payload_bits = text_to_bits(message)
    payload_syms = bpsk_mod(payload_bits)
    train_syms = np.ones(len(active), dtype=np.complex128)

    x_train, _, _ = ofdm_modulate(train_syms, n_fft=n_fft, n_cp=n_cp, active=active)
    x_payload, n_payload_ofdm, n_payload_syms = ofdm_modulate(payload_syms, n_fft=n_fft, n_cp=n_cp, active=active)
    x_bb = np.concatenate([x_train, x_payload])

    n = np.arange(len(x_bb))
    tx_audio = np.real(x_bb * np.exp(1j * 2 * np.pi * fc * n / fs))
    tx_audio = tx_audio / (np.max(np.abs(tx_audio)) + 1e-12)

    # Simulated acoustic path: direct sound plus echoes and additive noise.
    h_audio = np.zeros(180)
    h_audio[0] = 1.0
    h_audio[31] = 0.45
    h_audio[87] = -0.30
    h_audio[142] = 0.18
    h_audio = h_audio / np.sqrt(np.sum(h_audio**2))
    rx_audio = apply_channel(tx_audio, h_audio, keep_len=True)
    rx_audio = add_awgn(rx_audio, 28.0, rng)

    write_wav(AUDIO_DIR / "tx_packet.wav", tx_audio, fs)
    write_wav(AUDIO_DIR / "rx_loopback.wav", np.real(rx_audio), fs)

    # Downconvert and low-pass. The factor of 2 corrects for real passband mixing.
    rx_mixed = 2.0 * rx_audio * np.exp(-1j * 2 * np.pi * fc * n / fs)
    rx_bb = fft_lowpass(rx_mixed, fs=fs, cutoff_hz=3000.0)

    # Demodulate with known packet start. First block is a training symbol.
    n_total = n_payload_ofdm + 1
    raw = ofdm_demodulate(rx_bb, n_total, n_fft=n_fft, n_cp=n_cp, active=active, h_freq=None)
    raw_blocks = raw.reshape(n_total, len(active))
    H_est = raw_blocks[0] / train_syms
    payload_eq = (raw_blocks[1:] / np.where(np.abs(H_est) < 1e-12, 1e-12, H_est)).reshape(-1)
    bits_hat = bpsk_demod(payload_eq[:n_payload_syms])[: len(payload_bits)]
    decoded = bits_to_text(bits_hat)
    ber = bit_error_rate(payload_bits, bits_hat)

    # Spectrum figure of the transmitted audio packet.
    n_win = min(len(tx_audio), 8192)
    S = np.fft.rfft(tx_audio[:n_win] * np.hanning(n_win), 16384)
    f = np.fft.rfftfreq(16384, d=1 / fs)
    mag = 20 * np.log10(np.abs(S) / (np.max(np.abs(S)) + 1e-12) + 1e-12)
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(f, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, normalized)")
    plt.title("Real passband OFDM audio packet spectrum")
    plt.ylim([-90, 5])
    plt.grid(True, alpha=0.3)
    savefig("passband_audio_spectrum")

    with open(RES_DIR / "passband_audio_demo.txt", "w") as ftxt:
        ftxt.write("Passband audio OFDM synthetic loopback\n")
        ftxt.write(f"Carrier frequency: {fc:.1f} Hz\n")
        ftxt.write(f"Sample rate: {fs} Hz\n")
        ftxt.write(f"Original message: {message}\n")
        ftxt.write(f"Decoded message:  {decoded}\n")
        ftxt.write(f"BER: {ber:.6g}\n")
        ftxt.write("Limitation: packet timing is assumed known; add correlation-based synchronization for a true speaker/microphone test.\n")

    print("Wrote audio loopback outputs to", AUDIO_DIR)
    print("Decoded:", decoded)
    print("BER:", ber)


if __name__ == "__main__":
    main()
