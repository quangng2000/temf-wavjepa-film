# WavJEPA-Nat conditioning contract

Status: selected for the future lab on 2026-10-19 and wired into its experimental
Lightning/Hydra trainer. CPU tests use mock Nat outputs; actual pretrained extraction
has not been run in this scaffold. This does not migrate the existing RMFSR training
run or establish an audio-quality benefit. See the [training guide](training.md).

## Pinned source

- Model: `labhamlet/wavjepa-nat-base`.
- Model and processor revision: `15d95ff67fa98117b17e83a1653bbca97877ff6f`.
- Sample rate: 16,000 Hz; upstream input shape `[batch, 2, samples]`.
- Use the released context-encoder inference path, with parameters frozen and
  evaluation mode enabled. The JEPA predictor and EMA teacher are not FiLM heads.
- Review the pinned custom Python code before loading with `trust_remote_code`.

## Mono restoration baseline

The clean target and degraded STFT branches remain mono. Duplicate only the observed
waveform for the Nat branch: `[observed, observed]`. The paper uses duplication to
evaluate Nat on mono recordings. This supplies no additional spatial information.
Do not put the clean target, estimated noise, or Gaussian flow prior in channel two.

Use equal-length crops per batch. The pinned feature extractor accepts a list of
two-channel arrays, applies a shared RMS gain to -14 dBFS per example, and leaves
exact silence unchanged. Its variable-length padding must be tested before use;
passing an arbitrary padded batch without valid-length handling is not specified.
Encoder-only normalization must not change the clean/observed STFT amplitudes.

The model applies segment normalization internally, jointly across channels and
samples, and runs on 2.01-second segments. Keep this behavior identical in training
and inference. This is not a causal or zero-lookahead feature extractor.

## Features and FiLM

The pinned model returns `(features, timestamps_ms)`:

- Features: `[batch, 2, encoder_frame, 768]`.
- Timestamps: `[batch, encoder_frame]`, expressed in milliseconds.
- Lab choice: concatenate left/right embeddings along the feature dimension at the
  same time position, giving `[batch, encoder_frame, 1536]`.
- Align to the configured STFT frame times in seconds before trainable adapters.
  Never flatten the channel axis into the time axis or globally pool over time.

Concatenation preserves both learned channel branches, even for duplicated mono.
It is a design choice for this lab, not a separation guarantee. Adapters map 1536
features into each selected block's FiLM width; their final scale/shift heads start
at zero so modulation starts as the identity.

The upstream timestamp helper distributes times over the recording duration.
Treat these as the released timestamp convention, not a verified convolution-center
mapping. Check impulses and segment boundaries before choosing the final alignment.
The upstream model uses `inference_mode`; clone detached outputs outside that mode
before training adapters so autograd can save ordinary input tensors.

## Reuse and implementation checks

Cache identity must include model/revision, degraded-waveform hash, sample rate,
channel policy, preprocessing, segmentation, feature concatenation, and alignment.
Changing any of these invalidates the cache. Existing mono wavJEPA caches and
trained 768-input adapters are incompatible with this 1536-feature contract.

Before a training experiment, check finite features on silence and short clips,
segment boundaries, unchanged STFT inputs, deterministic repeated extraction,
frame/channel ordering, and nonzero adapter gradients with zero encoder gradients.
Measure frozen inference parameters and full loaded checkpoint parameters separately;
the latter includes pretraining modules. No exact combined parameter count is claimed.

Keep the full TEMF objective, sampler, JVP, and trainable restoration scope unchanged.
The frozen features depend only on observed audio and stay fixed along the flow JVP.

## References

- [Paper: Nat architecture and mono evaluation, §3.3–3.4](https://arxiv.org/html/2509.23238v1#S3.SS3)
- [Pinned model and processor files](https://huggingface.co/labhamlet/wavjepa-nat-base/tree/15d95ff67fa98117b17e83a1653bbca97877ff6f)
- [Released inference and output layout](https://huggingface.co/labhamlet/wavjepa-nat-base/blob/15d95ff67fa98117b17e83a1653bbca97877ff6f/model.py)
- [Preprocessing](https://huggingface.co/labhamlet/wavjepa-nat-base/blob/15d95ff67fa98117b17e83a1653bbca97877ff6f/feature_extraction_wavjepa_nat.py)
- [Normalization and timestamps](https://huggingface.co/labhamlet/wavjepa-nat-base/blob/15d95ff67fa98117b17e83a1653bbca97877ff6f/utils.py)
