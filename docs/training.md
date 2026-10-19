# Train, evaluate and restore speech

For the model equations and training/inference pseudocode, see
[Training and sampling explained](training-and-sampling.md).

For entrypoints, config groups and output locations, see
[Project structure and configuration](project-structure.md). The lab is independent
of the existing RMFSR training process.

## Install and smoke-test

Python 3.11 or 3.12 is required. From this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,augment]'
python -m src.upstream.fetch
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m src.train experiment=smoke
```

The fetch command caches pinned, checksum-verified TEMF source, without weights.
The smoke experiment performs two optimizer updates with synthetic signals, a tiny
network and a synthetic conditioner on CPU. It downloads nothing. Real training
requires explicit manifests and the pinned Nat weights.

## Real paired data and Nat weights

Provide separate train/validation JSONL files with one aligned pair per line:

```json
{"id":"clip-001","group":"speaker-001","clean":"clean.wav","observed":"damaged.wav"}
```

Audio paths resolve relative to the manifest. Both files must be finite 16 kHz mono,
equal length and at least four seconds under the default crop. The same crop is
used for both. Validation uses the center crop; training crops randomly. No
automatic loudness normalization, resampling or new degradation is applied.
For generated training pairs, select the explicit [augmentation mode](augmentation.md).
The default paired mode still uses your supplied observations without extra effects.
Caller-provided `group` identifies the speaker/source split;
IDs, groups and paths must not overlap between manifests. This cannot detect
incorrect group labels or hidden duplicates under different filenames.

Cache the pinned `labhamlet/wavjepa-nat-base` revision
`15d95ff67fa98117b17e83a1653bbca97877ff6f` using Hugging Face, or explicitly set
`model.conditioner.local_files_only=false` when intending a download. Loading uses
the pinned custom model code. See [the encoder contract](wavjepa-nat.md).

## Training and resume

```sh
python -m src.train trainer=gpu data.train_manifest=/absolute/train.jsonl \
  data.val_manifest=/absolute/val.jsonl

python -m src.train trainer=gpu data.train_manifest=/absolute/train.jsonl \
  data.val_manifest=/absolute/val.jsonl \
  ckpt_path=/absolute/checkpoints/last.ckpt
```

Use `trainer=cpu` for CPU execution. Both presets use FP32 and one device;
the GPU preset is configuration support, not a measured performance result.

Schedules use completed optimizer updates toward 300,000, including after resume.
Checkpoints are saved every 1,000 updates; validation every 5,000. EMA is updated
once per optimizer step via PyTorch `AveragedModel`, then checkpointed.

Resume checks the model/encoder/transform/objective recipe. Prepared and generated
training datasets use TorchData StatefulDataLoader; Lightning saves its iteration
state. Dataset RNG is checkpointed for crops and augmentation seeds. Replay tests
cover workers=0/2 and a single-device CPU Lightning resume. Keep worker count,
dataset files and recipe unchanged; old checkpoints without loader state cannot
reproduce data order. File identity uses paths/sizes/mtimes, not a full corpus hash.
Synthetic smoke loading remains ordinary DataLoader. Earlier lab U-Net/EMA and
RMFSR checkpoints are incompatible and rejected.

## Evaluate a checkpoint

```sh
python -m src.eval ckpt_path=/absolute/checkpoints/last.ckpt \
  data.val_manifest=/absolute/val.jsonl

python -m src.eval ckpt_path=/absolute/checkpoints/last.ckpt \
  data.val_manifest=/absolute/val.jsonl weights=ema
```

Evaluation runs on the supplied validation manifest. It does not create a test
split or reuse training data automatically. Raw weights are the default;
`weights=ema` selects the checkpoint's EMA weights. To evaluate a smoke checkpoint,
add `experiment=smoke` to select synthetic data. The model configuration always
comes from the checkpoint.

Validation reports NFE-1 coefficient MSE and **TorchMetrics STOI** for restored and
input audio. STOI runs on CPU through `pystoi` and needs adequate active reference
speech. Smoke runs disable STOI because their 64-ms synthetic signals are too short.

## Restore a recording

```sh
python -m src.sample checkpoint=/absolute/checkpoints/last.ckpt \
  input=/absolute/damaged.wav output=/absolute/restored.wav weights=ema nfe=20
```

Use only trusted lab checkpoints. The sampler reuses Nat context across NFE steps,
saves an unnormalized float WAV, and records peak/RMS, step, seed and weight choice.
It rejects existing output paths. GPU memory/performance, real pretrained Nat
alignment, multi-GPU/mixed precision and held-out restoration quality still need
measurement before any funded run. Only FP32 single-device CPU smoke execution is
validated in this scaffold.

## Logs and checks

CSV logs record weighted/raw losses, target/correction RMS, temporal weights,
boundary fractions, pre-clipping gradient norm and clipping indicators. Nonfinite
values reject an update with an exception and failure record. Gradient clipping
defaults to 1.0; there is no automatic early stopping or cloud-resource management.
The [algorithm guide](training-and-sampling.md) records the model assumptions,
objective, derivative calculations and limits of the checks.
