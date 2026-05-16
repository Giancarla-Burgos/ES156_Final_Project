#!/usr/bin/env python3
"""Create audio/tx_packet.wav for optional speaker/microphone testing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_ofdm.audio_utils import AudioConfig, encode_packet, write_wav  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="ES156 acoustic OFDM demo")
    parser.add_argument("--out", default=str(ROOT / "audio" / "tx_packet.wav"))
    args = parser.parse_args()
    cfg = AudioConfig()
    x, meta = encode_packet(args.message, cfg)
    write_wav(args.out, x, cfg.fs)
    print(f"Wrote {args.out}")
    print(meta)


if __name__ == "__main__":
    main()
