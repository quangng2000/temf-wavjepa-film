# Library-backed paired audio

This opt-in baseline uses audiomentations 0.43.1, pyroomacoustics 0.10.0,
fast-mp3-augment 0.2.0 and SciPy. No noise corpus, speech corpus or model weights
are downloaded. Install `python -m pip install -e '.[dev,augment]'` in the lab venv.

## Explicit inputs

Training speech JSONL has one file per row:
```json
{"id":"clip-001","group":"speaker-001","clean":"speech/clip-001.wav"}
```
Noise JSONL uses `path`:
```json
{"id":"noise-001","group":"recording-001","path":"noise/noise-001.wav"}
```
Paths resolve relative to each manifest. Speech/noise must be 16 kHz mono and
at least crop length. Training checks headers, not complete corpus integrity.
Validation stays a fixed clean/observed manifest described in [training](training.md).
Speech IDs/groups/paths cannot overlap. Select noise recordings that are separate
from those used to create validation: hidden overlap inside prepared mixtures is
not detectable from waveform filenames. Noise paths cannot be speech/validation paths.

Create a small reusable room bank in a new directory:
```sh
python -m src.augmentation.prepare_rooms datasets/rooms-train --count 32 --seed 42
```
This runs room simulation only, then exits; it refuses to overwrite a directory.
Each room has direct-only and full RIRs from the same geometry. The generated
`rooms.jsonl` references NPZ filters and their geometry/seed metadata. Training
loads these saved filters, so room simulation is not performed per optimizer update.

Select generated pairs explicitly:
```sh
python -m src.train data=augmented \
  data.train_manifest=/absolute/clean-train.jsonl \
  data.val_manifest=/absolute/fixed-validation.jsonl \
  data.augmentation.noise_manifest=/absolute/noise-train.jsonl \
  data.augmentation.rooms_manifest=/absolute/rooms-train/rooms.jsonl
```
The command retains the default CPU device. Select a GPU only when intentionally
starting a separately authorized lab run. Nat weights must already be cached.

## Signal path and defaults

1. Read a random clean crop with up to one second of real surrounding context.
2. With probability 0.5, apply matched direct/full room filters using SciPy FFT
   convolution. A common delay from the direct RIR aligns both branches; a common
   direct-RIR energy scale preserves their relative gain. Target is direct-path
   speech; observation includes reflections. Without room processing, target is dry.
3. Crop both branches identically, then add recorded noise to the observation
   with probability 0.8 at uniformly sampled 10–30 dB **whole-crop RMS** SNR.
4. Apply library high/low-pass filters (0.25 each), notch, clipping, tanh distortion
   and time mask (0.1 each), MP3 (0.2), and observation gain (0.3), in that order.
   Filters use zero phase. Default time masks span 10–80 ms for four-second crops.
5. MP3 uses the fast in-memory backend with `preserve_delay=False`. Gain is −6 to
   +3 dB. Joint attenuation prevents peaks exceeding 0.95 while preserving the
   clean/observed gain relation; targets are never normalized independently.

All probabilities/ranges are visible in `configs/data/augmented.yaml`. Effects
compose; the configured noise SNR does not describe the final signal after every
effect. These moderate defaults are engineering choices, not validated optimal
settings or the paper's distribution. There is one speaker per example and no
studio EQ/compression target, GSM, spectral bubble masking or exact paper parity.
Default rooms use shoebox geometry, absorption 0.3–0.7 and image-source order 6;
there is no claim of a specified RT60. More room diversity requires an explicit bank.

## Reproducibility and checks

Each example records its ID, crop offset and augmentation seed. Python/NumPy RNG
state used by library transforms is scoped to that example and restored afterward.
Worker-local dataset RNG plus loader state supports replay through Lightning.
Use the in-training ModelCheckpoint files: a new manual checkpoint saved after
Trainer teardown does not retain the original iterator cursor in this integration.
Dataset/recipe identity includes paths, file sizes/mtimes, settings and library
versions; this is lightweight change detection, not content integrity certification.

Tests exercise real library effects on synthetic signals: SNR, common room arrival,
MP3 delay, target preservation, deterministic seeds, silence and peak handling,
fixed validation, worker replay and Lightning checkpoint resume. These tests do not
establish real-speech quality, corpus readiness, DDP behavior or GPU throughput.
The model objective, conditioning, optimizer and TEMF schedules are unchanged.
