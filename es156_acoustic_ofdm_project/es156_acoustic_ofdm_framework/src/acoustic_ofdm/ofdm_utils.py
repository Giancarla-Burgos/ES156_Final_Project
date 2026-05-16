"""Small OFDM and PAM utilities for an ES 156 final project.

The code is intentionally lightweight: numpy does the signal processing and
matplotlib only appears in the experiment script. The implementation uses a
complex baseband OFDM model for the synthetic experiments. That keeps the math
close to the course concepts: convolution, DFT/FFT, cyclic prefix, and one-tap
frequency-domain equalization.
"""

from __future__ import annotations

import numpy as np


def rng_from_seed(seed: int = 156) -> np.random.Generator:
    """Return a reproducible random number generator."""
    return np.random.default_rng(seed)


def bpsk_mod(bits: np.ndarray) -> np.ndarray:
    """Map bits {0,1} to BPSK symbols {+1,-1}."""
    b = np.asarray(bits).astype(int).ravel()
    return (1.0 - 2.0 * b).astype(np.complex128)


def bpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision BPSK demodulation using the real axis."""
    return (np.real(symbols) < 0.0).astype(np.uint8)


def text_to_bits(text: str) -> np.ndarray:
    """Convert UTF-8 text to a big-endian bit array."""
    data = text.encode("utf-8")
    raw = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(raw, bitorder="big")


def bits_to_text(bits: np.ndarray) -> str:
    """Convert a big-endian bit array back to UTF-8 text."""
    b = np.asarray(bits).astype(np.uint8).ravel()
    n = (len(b) // 8) * 8
    if n == 0:
        return ""
    raw = np.packbits(b[:n], bitorder="big")
    return bytes(raw.tolist()).decode("utf-8", errors="replace")


def default_active_subcarriers(n_fft: int = 64) -> np.ndarray:
    """Return 52 active subcarriers in a simplified 64-point OFDM system.

    The returned values are signed frequency-bin numbers: -26,...,-1,+1,...,+26.
    Bin 0 is left unused as a DC/center guard. This is inspired by 802.11-style
    OFDM, but this project is not trying to implement the full standard.
    """
    if n_fft < 64:
        raise ValueError("This helper assumes n_fft >= 64.")
    return np.r_[np.arange(-26, 0), np.arange(1, 27)]


def subcarrier_to_fft_index(k_signed: np.ndarray, n_fft: int) -> np.ndarray:
    """Map signed subcarrier labels to numpy FFT bin indices."""
    return np.mod(k_signed, n_fft).astype(int)


def ofdm_modulate(
    symbols: np.ndarray,
    n_fft: int = 64,
    n_cp: int = 16,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Modulate complex symbols into a concatenated OFDM waveform.

    Returns waveform, number of OFDM symbols, and number of payload symbols.
    """
    if active is None:
        active = default_active_subcarriers(n_fft)
    active = np.asarray(active, dtype=int)
    bins = subcarrier_to_fft_index(active, n_fft)

    s = np.asarray(symbols, dtype=np.complex128).ravel()
    n_payload_symbols = len(s)
    n_active = len(active)
    n_symbols = int(np.ceil(n_payload_symbols / n_active))
    n_pad = n_symbols * n_active - n_payload_symbols
    if n_pad > 0:
        s = np.concatenate([s, np.ones(n_pad, dtype=np.complex128)])

    S = s.reshape(n_symbols, n_active)
    X = np.zeros((n_symbols, n_fft), dtype=np.complex128)
    X[:, bins] = S

    # numpy's IFFT contains 1/N. Multiplying by sqrt(N) keeps energy convenient.
    x_blocks = np.fft.ifft(X, axis=1) * np.sqrt(n_fft)
    if n_cp > 0:
        x_blocks = np.concatenate([x_blocks[:, -n_cp:], x_blocks], axis=1)
    return x_blocks.reshape(-1), n_symbols, n_payload_symbols


def ofdm_demodulate(
    waveform: np.ndarray,
    n_symbols: int,
    n_fft: int = 64,
    n_cp: int = 16,
    active: np.ndarray | None = None,
    h_freq: np.ndarray | None = None,
) -> np.ndarray:
    """Demodulate a concatenated OFDM waveform into complex subcarrier symbols."""
    if active is None:
        active = default_active_subcarriers(n_fft)
    active = np.asarray(active, dtype=int)
    bins = subcarrier_to_fft_index(active, n_fft)
    block_len = n_fft + n_cp
    needed = n_symbols * block_len
    y = np.asarray(waveform, dtype=np.complex128).ravel()
    if len(y) < needed:
        y = np.pad(y, (0, needed - len(y)))
    y = y[:needed].reshape(n_symbols, block_len)
    if n_cp > 0:
        y = y[:, n_cp:]
    Y = np.fft.fft(y, axis=1) / np.sqrt(n_fft)
    S_hat = Y[:, bins]
    if h_freq is not None:
        H = np.asarray(h_freq, dtype=np.complex128).reshape(1, -1)
        eps = 1e-12
        S_hat = S_hat / np.where(np.abs(H) < eps, eps, H)
    return S_hat.reshape(-1)


def apply_channel(x: np.ndarray, h: np.ndarray, keep_len: bool = True) -> np.ndarray:
    """Apply an FIR channel by linear convolution."""
    y = np.convolve(np.asarray(x), np.asarray(h), mode="full")
    if keep_len:
        return y[: len(x)]
    return y


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add real or complex AWGN at a specified signal-to-noise ratio in dB."""
    x = np.asarray(x)
    power = np.mean(np.abs(x) ** 2)
    if power <= 0:
        raise ValueError("Cannot add AWGN to a zero-power signal.")
    noise_power = power / (10.0 ** (snr_db / 10.0))
    if np.iscomplexobj(x):
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )
    else:
        noise = np.sqrt(noise_power) * rng.standard_normal(x.shape)
    return x + noise


def multipath_channel() -> np.ndarray:
    """A short synthetic frequency-selective complex baseband channel."""
    h = np.zeros(15, dtype=np.complex128)
    h[0] = 1.0
    h[3] = 0.55 * np.exp(1j * 0.45)
    h[8] = 0.35 * np.exp(-1j * 0.90)
    h[14] = 0.22 * np.exp(1j * 1.30)
    h = h / np.sqrt(np.sum(np.abs(h) ** 2))
    return h


def channel_frequency_response(h: np.ndarray, n_fft: int, active: np.ndarray | None = None) -> np.ndarray:
    """Return the channel DFT samples on the active OFDM bins."""
    if active is None:
        active = default_active_subcarriers(n_fft)
    H = np.fft.fft(h, n_fft)
    return H[subcarrier_to_fft_index(active, n_fft)]


def pam_transmit(symbols: np.ndarray, sps: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Generate a rectangular-pulse single-carrier PAM/BPSK waveform."""
    pulse = np.ones(sps, dtype=float) / np.sqrt(sps)
    up = np.zeros(len(symbols) * sps, dtype=float)
    up[::sps] = np.real(symbols)
    x = np.convolve(up, pulse, mode="full")
    return x, pulse


def pam_matched_filter_detect(y: np.ndarray, pulse: np.ndarray, n_symbols: int, sps: int = 8) -> np.ndarray:
    """Matched-filter receiver for rectangular-pulse PAM/BPSK."""
    mf = pulse[::-1]
    r = np.convolve(np.real(y), mf, mode="full")
    sample_start = len(pulse) - 1
    samples = r[sample_start : sample_start + n_symbols * sps : sps]
    if len(samples) < n_symbols:
        samples = np.pad(samples, (0, n_symbols - len(samples)))
    return bpsk_demod(samples[:n_symbols])


def bit_error_rate(bits: np.ndarray, bits_hat: np.ndarray) -> float:
    """Compute BER between equal-length prefixes of two bit arrays."""
    n = min(len(bits), len(bits_hat))
    if n == 0:
        return np.nan
    return float(np.mean(np.asarray(bits[:n]) != np.asarray(bits_hat[:n])))
