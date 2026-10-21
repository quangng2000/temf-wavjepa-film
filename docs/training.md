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

### 98M model with effective batch eight

The opt-in `experiment=temf_100m_accum8` preset uses 98,179,238 trainable parameters,
FP32, and the frozen Nat encoder. Supply the four manifests described in
[augmentation mode](augmentation.md) when selecting it. It does not change defaults.

The loader supplies eight clips. Before encoding, the objective samples all eight
time pairs, noise tensors, and temporal weights together. Training then encodes
two clips at a time and backpropagates each microbatch mean multiplied by `2/8`.
Only after all four groups does it clip the accumulated gradient and advance AdamW
and EMA once. The scheduler advances once per effective batch, including on resume.
Do not also enable Lightning's `accumulate_grad_batches`; it remains `1`.

The pinned TEMF forward keeps its full continuous JVP target, velocity mixture,
temporal weighting, and boundary scheduler. A checked adapter only replaces its
three sampling/weight assignments with prepared full-batch values. This preserves
the floor-based boundary count: seven diagonal examples initially and one near the
end of the 300,000-update schedule. Sampling four independent batches of two would
lose that late diagonal example. The microbatch path requires single-device FP32
continuous TEMF; validation retains the normal validation batch handling.

A four-update A100 SXM 80 GB test on real four-second clips passed at schedule
positions 0, 1, 299998, and 299999. Peak allocated memory was 50.04 GiB; allocator
reservation peaked at 66.83 GiB. Late-schedule updates took 29.3–29.9 seconds per
effective batch. These are short capacity/correctness checks from fresh weights,
not evidence of convergence or long-run stability. Production was not started.

### Faster, equivalent forward-mode JVP

Select `experiment=temf_100m_accum8 model=temf_100m_forward_ad` to compute the
full TEMF directional derivative with PyTorch forward-mode AD. The default still
uses the authors' `torch.autograd.functional.jvp`, which computes a JVP through
double backward. The opt-in backend keeps the state and time derivative terms,
detached target, velocity mixture, temporal weights and boundary schedule. It
does not use finite differences or remove the JVP correction. Both presets use
the same FP32 settings, effective batch eight and microbatch two.

On the A100 SXM 80 GB, a matched two-update late-schedule probe measured 29.73
seconds for the warmed-up reverse backend versus 4.05 seconds for forward AD.
The JVP target stage fell from 27.35 to 1.74 seconds. Peak allocated memory fell
from 50.02 to 15.68 GiB. The configured backend also completed two early-schedule
updates; its warmed-up update took 2.67 seconds. The first update includes startup
overhead and took 6.08 seconds late or 4.75 seconds early. These short probes used
fresh weights and a small set of real four-second clips, not production training.

A separate full-model comparison used identical weights, data and random draws.
Corrected targets differed by at most 5.96e-8, and all 98,179,238 parameter-gradient
values passed `atol=1e-5, rtol=1e-4` (maximum absolute difference 5.59e-9).
The full derivative, detached target and loss were also checked. Floating-point
rounding differs between AD algorithms; these checks do not establish long-run
stability, convergence or identical training trajectories.

The backend is recorded in the objective recipe, so switching an existing
checkpoint requires an explicit migration rather than silently changing its
numerics. Performance probes and equivalence reports are stored locally under
`runs/jvp-speed-20261019/`; no production controller is started by these checks.

### Opt-in normalized TrigFlow path

To test a variance-preserving trigonometric path with the same 325M network,
forward-mode full TEMF derivative, effective batch eight, and 200,000-update
schedule, select:

```sh
python -m src.train experiment=temf_325m_trigflow_accum8 \
  data.train_manifest=/absolute/train.jsonl \
  data.val_manifest=/absolute/val.jsonl
```

This changes the clean-to-noise path and analytic path velocity. It does not
change the TEMF scheduler, weighting, target, JVP, or solver time coordinate.
It is an experiment from fresh weights; the changed checkpoint recipe prevents
silently resuming a linear-path checkpoint. See the equations and scope in
[Training and sampling explained](training-and-sampling.md#opt-in-normalized-trigflow-path).

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
