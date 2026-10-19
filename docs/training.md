# Training scaffold

The executable lab follows the structure of
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144):
`src/train.py`, a LightningModule/DataModule, Hydra config groups and experiment
overrides. Implementation is specific to this lab; no MNIST trainer is included.
It does not connect to or control the existing RMFSR Runpod process.

## Install and smoke-test

Python 3.11 or 3.12 is required. From this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m src.train experiment=smoke
```

The smoke experiment performs two optimizer updates on synthetic signals and a
synthetic conditioner. It downloads nothing and uses CPU. It validates execution,
not speech quality or the pretrained Nat checkpoint. The default training config
instead requires explicit real manifests and cached Nat weights; it fails without them.

## Real paired data and Nat weights

Provide separate train/validation JSONL files with one aligned pair per line:

```json
{"id":"clip-001","group":"speaker-001","clean":"clean.wav","observed":"damaged.wav"}
```

Paths resolve relative to the manifest. Both files must be finite 16 kHz mono,
equal length and at least four seconds under the default crop. The same crop is
used for both. Validation uses the center crop; training crops randomly. No
automatic loudness normalization, resampling or new degradation is applied.
Supply already augmented pairs; the paper's on-the-fly augmentation generator is
not implemented here. Caller-provided `group` identifies the speaker/source split;
IDs, groups and paths must not overlap between manifests. This cannot detect
incorrect group labels or hidden duplicates under different filenames.

Cache the pinned `labhamlet/wavjepa-nat-base` revision
`15d95ff67fa98117b17e83a1653bbca97877ff6f` using Hugging Face, or explicitly set
`model.conditioner.local_files_only=false` when intending a download. Loading uses
the pinned custom model code. See [the encoder contract](wavjepa-nat.md).

## Training and resume

```sh
python -m src.train data.train_manifest=/absolute/train.jsonl \
  data.val_manifest=/absolute/val.jsonl trainer.accelerator=gpu

python -m src.train data.train_manifest=/absolute/train.jsonl \
  data.val_manifest=/absolute/val.jsonl trainer.accelerator=gpu \
  ckpt_path=/absolute/checkpoints/last.ckpt
```

The proposed network is a frequency-downsampling residual U-Net, with frame-local
normalization, positional time controls, and zero-initialized Nat FiLM heads.
Its **303,323,650 trainable parameters** exclude frozen Nat. This architecture
and AdamW settings are lab baselines, not the TEMF authors' image model.

The full FP32 continuous TEMF objective uses direct velocity, a unit Gaussian prior,
the released boundary/mix schedules, interval weighting and full state/time JVP.
Observed STFT/context remain fixed along that JVP. There is no CFG, adaptive loss
rescaling, target clipping, derivative removal or auxiliary speech-fidelity loss.

Schedules use completed optimizer updates toward 300,000, including after resume.
Checkpoints are saved every 1,000 updates; validation every 5,000. EMA is updated
once per optimizer step and saved alongside raw weights and optimizer state.
Validation currently reports **raw-weight NFE-1 coefficient MSE**, plus the input
baseline. These are not intelligibility or perceptual-quality metrics.

CSV logs contain weighted/raw losses, target/JVP/correction RMS, temporal weights,
boundary fractions, pre-clipping gradient norm and clipping indicator. Nonfinite
values reject an update with an exception and failure record. Gradient clipping
is 1.0. There is no pod lifecycle code and no automatic early stopping.

Resume checks the model/encoder/transform/objective recipe. Mid-epoch data order and
random crops are not bit-for-bit replayable; optimizer/EMA/global-step restoration
is tested. Old RMFSR coefficient-prediction checkpoints cannot be resumed here.

## Sampling

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
