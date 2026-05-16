"""Optional real-valued audio OFDM utilities.

The main paper can be completed using only scripts/run_experiments.py.  This file is
an extension for a speaker/microphone or simulated audio-loopback demonstration.
It uses Hermitian symmetry in the DFT so that the IFFT output is real-valued and
can be written directly as a WAV audio signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from .ofdm_utils import add_awgn, bpsk_mod, bpsk_demod, text_to_bits, bits_to_text


@dataclass(frozen=True)
class AudioConfig:
    fs: int = 48_000
    n_fft: int = 256
    n_cp: int = 64
    silence_s: float = 0.25
    preamble_repetitions: int = 2

    @property
    def active_pos(self) -> np.ndarray:
        # Positive bins only. At fs=48 kHz, k=16...64 is about 3-12 kHz.
        return np.arange(16, 65)

    @property
    def bits_per_symbol(self) -> int:
        return len(self.active_pos)  # BPSK, one bit per active positive subcarrier

    @property
    def block_len(self) -> int:
        return self.n_fft + self.n_cp


def write_wav(path: str | Path, x: np.ndarray, fs: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 0:
        x = 0.95 * x / peak
    pcm = np.asarray(np.clip(x, -1.0, 1.0) * 32767, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        fs = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError("This simple reader expects 16-bit PCM WAV files.")
    x = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return fs, x


def _hermitian_ifft_block(pos_symbols: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    X = np.zeros(cfg.n_fft, dtype=np.complex128)
    pos = cfg.active_pos
    X[pos] = pos_symbols
    X[-pos] = np.conj(pos_symbols)
    x = np.fft.ifft(X) * np.sqrt(cfg.n_fft)
    x = np.real(x)
    return np.r_[x[-cfg.n_cp:], x]


def _fft_pos_from_block(block_with_cp: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    block = block_with_cp[cfg.n_cp:cfg.n_cp + cfg.n_fft]
    X = np.fft.fft(block) / np.sqrt(cfg.n_fft)
    return X[cfg.active_pos]


def make_preamble(cfg: AudioConfig) -> np.ndarray:
    # A deterministic all-ones BPSK pilot. Random pilots also work, but all-ones
    # makes the channel-estimation equation especially transparent.
    pilot = np.ones(len(cfg.active_pos), dtype=np.complex128)
    block = _hermitian_ifft_block(pilot, cfg)
    return np.tile(block, cfg.preamble_repetitions)


def _length_bits(n_bits: int) -> np.ndarray:
    if n_bits >= 2**16:
        raise ValueError("message too long for the 16-bit demo header")
    return np.array([(n_bits >> k) & 1 for k in range(15, -1, -1)], dtype=np.uint8)


def _parse_length(bits: np.ndarray) -> int:
    value = 0
    for b in bits[:16].astype(int):
        value = (value << 1) | b
    return value


def encode_packet(message: str, cfg: AudioConfig) -> tuple[np.ndarray, dict]:
    payload = text_to_bits(message)
    bits = np.r_[_length_bits(len(payload)), payload]
    pad = (-len(bits)) % cfg.bits_per_symbol
    if pad:
        bits = np.r_[bits, np.zeros(pad, dtype=np.uint8)]
    symbols = bpsk_mod(bits).reshape(-1, len(cfg.active_pos))
    data_blocks = [_hermitian_ifft_block(row, cfg) for row in symbols]
    data = np.concatenate(data_blocks) if data_blocks else np.array([], dtype=float)
    silence = np.zeros(int(round(cfg.silence_s * cfg.fs)))
    packet = np.r_[silence, make_preamble(cfg), data, silence]
    packet = 0.75 * packet / max(1e-12, np.max(np.abs(packet)))
    meta = {
        "payload_bits": int(len(payload)),
        "data_blocks": int(len(data_blocks)),
        "pad_bits": int(pad),
        "duration_s": float(len(packet) / cfg.fs),
    }
    return packet, meta


def find_start(recording: np.ndarray, cfg: AudioConfig) -> tuple[int, np.ndarray]:
    pre = make_preamble(cfg)
    x = np.asarray(recording, dtype=float)
    corr = np.correlate(x, pre, mode="valid")
    energy = np.convolve(x**2, np.ones(len(pre)), mode="valid")
    score = corr / np.maximum(1e-12, np.sqrt(energy * np.sum(pre**2)))
    return int(np.argmax(np.abs(score))), score


def decode_packet(recording: np.ndarray, cfg: AudioConfig, max_data_blocks: int | None = None) -> tuple[str, dict]:
    start, score = find_start(recording, cfg)
    pre_len = cfg.preamble_repetitions * cfg.block_len
    pilot = np.ones(len(cfg.active_pos), dtype=np.complex128)

    pilot_start = start + (cfg.preamble_repetitions - 1) * cfg.block_len
    pilot_block = recording[pilot_start:pilot_start + cfg.block_len]
    if len(pilot_block) < cfg.block_len:
        return "", {"reason": "not enough samples for pilot", "start": start}
    Hhat = _fft_pos_from_block(pilot_block, cfg) / pilot

    data_start = start + pre_len
    possible = max(0, (len(recording) - data_start) // cfg.block_len)
    if max_data_blocks is not None:
        possible = min(possible, max_data_blocks)
    out_bits = []
    for i in range(possible):
        b0 = data_start + i * cfg.block_len
        block = recording[b0:b0 + cfg.block_len]
        Y = _fft_pos_from_block(block, cfg)
        S = Y / np.where(np.abs(Hhat) < 1e-9, 1e-9, Hhat)
        out_bits.append(bpsk_demod(S))
    bits = np.concatenate(out_bits) if out_bits else np.array([], dtype=np.uint8)
    if len(bits) < 16:
        return "", {"reason": "not enough decoded bits", "start": start}
    n_payload = _parse_length(bits[:16])
    text = bits_to_text(bits[16:16 + n_payload])
    diag = {
        "start": int(start),
        "corr_peak": float(np.max(np.abs(score))),
        "estimated_payload_bits": int(n_payload),
        "decoded_blocks": int(possible),
        "channel_mag_median": float(np.median(np.abs(Hhat))),
        "channel_mag_min": float(np.min(np.abs(Hhat))),
    }
    return text, diag


def synthetic_audio_channel(x: np.ndarray, rng: np.random.Generator, snr_db: float = 24.0) -> tuple[np.ndarray, np.ndarray]:
    # Simple bandlimited echo channel, meant to resemble a speaker-room-mic path.
    # Keep the synthetic loopback shorter than the cyclic prefix so the
    # circular-convolution model is valid. Real rooms often violate this, which
    # is a useful limitation to discuss separately.
    n = np.arange(17) - 8
    cutoff = 0.28
    lp = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
    lp = lp / np.sum(lp)
    echo = np.zeros(45)
    echo[0] = 1.0
    echo[11] = 0.25
    echo[27] = -0.16
    echo[44] = 0.08
    h = np.convolve(lp, echo)
    h = h / np.sqrt(np.sum(h**2))
    y = np.convolve(x, h, mode="full")[:len(x)]
    y = add_awgn(y, snr_db, rng)
    y = 0.85 * y / max(1e-12, np.max(np.abs(y)))
    return y, h
