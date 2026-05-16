# ES156 Acoustic OFDM Modem Framework

Project title:

**A Sonar-Inspired Acoustic OFDM Testbed for Reproducible Multipath Packet-Recovery Experiments**

This project implements a reproducible ES156 final project around a central question: when can cyclic-prefix OFDM turn a dispersive acoustic channel into independent FFT subchannels, and what rate/robustness tradeoff does that require?

The project connects convolution, sampling, modulation, matched filtering, FFTs, cyclic prefixes, frequency-domain equalization, and preamble-based audio packet synchronization. The framing is sonar-inspired, but the included experiments are intentionally a simplified terrestrial/simulated acoustic testbed rather than a claim of underwater deployment or a deployment-grade modem.

## Paper focus

The updated paper is organized around the following research question:

> How much can cyclic-prefix OFDM improve packet recovery in a dispersive acoustic channel, and what cyclic-prefix/equalization conditions are needed before the link behaves like independent subchannels?

The key evidence is:

- a cyclic-prefix model test showing when linear convolution becomes an FFT-diagonal circular-convolution model;
- a CP-length BER sweep showing the delay-spread versus overhead tradeoff;
- Monte Carlo BER curves comparing CP-OFDM, OFDM without CP, OFDM without equalization, and single-carrier matched filtering;
- constellation plots showing how one-tap equalization reverses subcarrier-specific attenuation and rotation;
- a simulated real-valued WAV packet demo with preamble synchronization and training-symbol channel estimation.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_experiments.py
python3 scripts/run_audio_loopback_demo.py
python3 scripts/run_passband_audio_loopback.py
```

The scripts write:

- `figures/*.pdf` and `figures/*.png` for the paper/Overleaf
- `results/ber_summary.csv` with one deterministic BER run
- `results/ber_summary_mc.csv` with 20-trial Monte Carlo BER means, standard errors, and 95% confidence intervals
- `results/cp_length_sweep.csv` with BER versus cyclic-prefix length at 18 dB SNR
- `results/decoded_message.txt` with a decoded text-message demo
- `results/audio_loopback_summary.txt` with the Hermitian real-audio OFDM packet output
- `results/passband_audio_demo.txt` with the passband audio loopback output
- `audio/tx_packet.wav`, `audio/rx_loopback.wav`, and `audio/rx_loopback_simulated.wav`

## Main experiments

1. Synthetic complex-baseband CP-OFDM through a finite multipath channel, treated as a delay-spread stress test.
2. Single-carrier PAM matched filtering in AWGN and in a multipath channel.
3. Monte Carlo BER comparison across SNR.
4. Cyclic-prefix-length sweep showing the transition near the channel memory `L-1` and the throughput cost of extra prefix samples.
5. Real-valued Hermitian OFDM WAV packet with simulated loopback, preamble correlation, and one-symbol channel estimation.
6. Optional passband loopback centered near 8 kHz.

## Suggested extensions / limitations

1. Record `audio/tx_packet.wav` through an actual speaker/microphone and decode it.
2. Replace BPSK with QPSK or 16-QAM and plot BER vs. SNR.
3. Add carrier-frequency offset and implement a correction method.
4. Add sample-rate drift and timing tracking.
5. Add error-correcting codes or repetition coding.
6. Compare zero-forcing equalization to MMSE equalization in channels with deeper spectral nulls.

## Files

- `finalproject.tex`: IEEEtran paper source.
- `finalproject.pdf`: compiled paper.
- `scripts/run_experiments.py`: main synthetic experiments, Monte Carlo BER curves, and CP sweep.
- `scripts/run_audio_loopback_demo.py`: Hermitian real-valued audio OFDM packet with preamble synchronization.
- `scripts/run_passband_audio_loopback.py`: optional real passband loopback.
- `scripts/make_audio_packet.py`: generate a WAV packet for external recording.
- `scripts/decode_recorded_packet.py`: decode a recorded 48 kHz 16-bit mono WAV packet.
- `src/acoustic_ofdm/ofdm_utils.py`: core signal-processing utilities.
- `src/acoustic_ofdm/audio_utils.py`: real-valued audio packet utilities.

## Related-work context

The paper connects the project to classic OFDM and synchronization references; underwater/sonar-inspired OFDM work on long multipath, Doppler, sparse channel estimation, and cyclic-prefix overhead; and airborne speaker-microphone systems such as Acoustic OFDM, near-ultrasonic consumer-device communication, and Dolphin.
