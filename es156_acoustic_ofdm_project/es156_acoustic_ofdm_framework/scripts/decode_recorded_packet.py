#!/usr/bin/env python3
"""Decode a 48 kHz 16-bit mono WAV recorded from the acoustic OFDM packet."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_ofdm.audio_utils import AudioConfig, decode_packet, read_wav  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--max-data-blocks", type=int, default=None)
    args = parser.parse_args()
    cfg = AudioConfig()
    fs, x = read_wav(args.wav)
    if fs != cfg.fs:
        raise SystemExit(f"Expected {cfg.fs} Hz WAV but got {fs} Hz. Record or resample at 48 kHz.")
    text, diag = decode_packet(x, cfg, max_data_blocks=args.max_data_blocks)
    print("Decoded text:", repr(text))
    print("Diagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
