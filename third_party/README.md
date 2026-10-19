# Reused implementations

## SGMSE NCSN++

Source: [sp-uhh/sgmse](https://github.com/sp-uhh/sgmse), commit
`1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e`.
The selected source is vendored in `src/vendor/sgmse`; `sgmse.json` records original
and installed SHA-256 values. `sgmse.patch` shows every modified upstream file.
The repository MIT license is in `SGMSE-LICENSE`. Google-authored files retain
Apache-2.0 headers; the license text is in `APACHE-2.0.txt`. Original source stays
intact rather than being rewritten to meet local function/file size conventions.

Adaptations outside the vendor tree:

- Two times use the original positional embedding helper, concatenated before the
  original time MLP. Mean velocity is the output meaning; no diffusion-score scaling.
- Nat FiLM follows residual `GroupNorm_1`. Heads start at zero; context is linearly
  resized after padding to match each audio-time resolution. Encoder stays frozen.
- PyTorch RMSNorm normalizes time-MLP output and residual time projections.
  This is inspired by stable conditioning, not an exact sCM implementation.
- Pad frequency/time to the downsampling factor and crop the final output.
- Default nf=256 is wider than upstream nf=128. Extra resolution attention is
  disabled; bottleneck attention, BigGAN residual blocks, skip paths and FIR remain.

The vendor diff adds optional embedding/context inputs, adjusts the registry import,
and selects the authors' `upfirdn2d_native` on all devices instead of JIT CUDA ops.
New namespace/package files avoid importing unrelated backbones. No GPU throughput
or large-model memory claim has been validated. This is a noncausal model.

## TEMF

Source: [yuanpengtu/TEMF](https://github.com/yuanpengtu/TEMF), commit
`78c531cca978a23113b262cd0b6bede2197b1916`.
No license file was present at this revision. Its source is not redistributed in
this repository: `python -m src.upstream.fetch` explicitly downloads four pinned
files into ignored `.cache/upstream/temf/`. Every file is SHA-256 verified before use.
Training never downloads or executes unpinned source automatically.

`src/upstream/selection.py` selects original class/method bodies with Python AST.
Bodies, including `TEMFLoss.forward`, target, weight and scheduling calculations,
are unchanged. Imports of the unrelated image stack are avoided; annotations are
deferred, the `beartype` decorator is removed, and the base class is `nn.Module`.
Upstream EasyDict and relevant misc helpers are also reused. The selected scope is
continuous alpha=0, no teacher, no image augmentation, no CFG/dropout, unit Gaussian
prior and unit branch weights. It is not a general adapter for every TEMF mode.

`AudioVelocity` inserts/removes a singleton video-frame axis and translates the
condition container. Nat/observed audio remain fixed during the original full JVP.
A target observer records RMS values without changing returned targets. Parameter
gradients and loss match an independent legacy reference at three training stages;
numeric schedule/weight/target fixtures remain checked. The reference rewrite lives
only under tests and is not called by production training.

## PyTorch and TorchMetrics

EMA uses `torch.optim.swa_utils.AveragedModel` and `get_ema_multi_avg_fn`, seeded
with initial parameters to retain the previous decay behavior from the first update.
A small Lightning callback handles timing and checkpoint lifecycle; PyTorch performs
all averaging. `avg_fn` is reconstructed on resume because it is not in state_dict.

STOI uses `torchmetrics.audio.ShortTimeObjectiveIntelligibility` backed by `pystoi`.
Input and restored signals have separate metric accumulators. Metrics are for raw
NFE-1 validation weights; sampling can independently select checkpointed EMA.

The pretrained WavJEPA-Nat loader and Lightning/Hydra infrastructure remain library
integrations. No production RMFSR files or Runpod processes are controlled here.

## Paired audio and resumable data loading

See [augmentation dependency versions, licenses and integration choices](augmentation.md).
The optional generator uses library DSP with a small coordinator for paired targets.
