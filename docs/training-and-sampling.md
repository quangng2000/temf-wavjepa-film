# Training and sampling

[Back to the project](../README.md) · [Setup and commands](training.md)

This document explains the experimental model and its training algorithm.
During training, aligned clean and degraded speech teach the model a velocity
field. During sampling, only the degraded recording is supplied: WavJEPA-Nat
provides context, and the solver uses the predicted velocity to move from a
Gaussian starting state toward restored speech.

The backbone is adapted from SGMSE NCSN++; TEMF executes pinned upstream code.
See [sources and adaptations](../third_party/README.md) for exact implementation
boundaries. CPU checks establish behavior, not full-model training stability or
audio quality.

## Contents

- [Mathematical checks](#mathematics-first)
- [Model scope](#agreed-scope)
- [Signal path](#implemented-signal-path)
- [Training and sampling pseudocode](#training-and-sampling-pseudocode)
- [TEMF fidelity requirements](#temf-fidelity-requirements)
- [Initialization and evaluation](#initialization-and-evaluation)
- [References](#references)

## Mathematics first

All formal proofs live in [`lean/`](../lean/README.md), with pinned Lean/Mathlib versions
and an axiom-auditing checker. They cover the MeanFlow identity, reverse solver,
TEMF weights/schedules, and FiLM derivatives. See [scope](../lean/SCOPE.md) for the
assumptions and limits. CPU tests compare TEMF weights, schedules, and the full
continuous target against a numeric reference from the pinned authors' code.
These checks establish implementation behavior, not restoration quality.

## Agreed scope

- Target approximately **300 million trainable restoration parameters**.
- Keep the selected **WavJEPA-Nat encoder frozen** in evaluation mode.
- Use WavJEPA-Nat features to condition restoration through trainable **FiLM** adapters.
- Predict the **mean velocity directly**, rather than converting a clean-coefficient prediction into velocity.
- Execute the pinned TEMF authors' objective and sampler through an audio interface.
- Reuse SGMSE NCSN++ layers and architecture, adding two-time controls and Nat FiLM.

Count the selected Nat inference path separately from the approximately 300 million
trainable restoration parameters. The former mono encoder's 89,280,768-parameter
count and combined 389M estimate do not establish the Nat model's count. The lab's
325,497,126 trainable parameters include its backbone and adapters; the upstream
checkpoint also contains pretraining modules unused by feature extraction.

## Implemented signal path

1. Encode degraded audio as compressed complex STFT coefficients.
2. Duplicate the same degraded mono waveform into two channels for frozen WavJEPA-Nat.
3. Concatenate its channel features at each timestamp, align to STFT audio frames,
   and project them through trainable adapters.
4. Generate FiLM scale and shift for selected restoration blocks.
5. Condition the restoration backbone on the flow state, degraded coefficients,
   current/destination flow times, and the aligned WavJEPA-Nat context.
6. Predict mean velocity and integrate it to the clean endpoint.
7. Decompress coefficients and reconstruct the waveform with inverse STFT.

FiLM starts as an identity modulation: `(1 + scale) * features + shift`,
with zero-initialized final scale/shift projections. The adapter applies it after
the second GroupNorm in every upstream residual block. Placement and normalization remain experimental choices;
they are not established quality improvements.

## Training and sampling pseudocode

This summarizes the **implemented experimental trainer**, not a validated production
run. It combines the pinned TEMF continuous target/weight/schedule with our speech
representation and conditioning design. See [`src/train.py`](../src/train.py) and the
[training guide](training.md). The animation runs a much smaller toy calculation;
it does not execute this full training loop.

For a concrete baseline, the pseudocode uses a **unit Gaussian prior**, compressed
complex STFT coefficients, conditional prediction without classifier-free guidance
or condition dropout, and **no auxiliary speech-fidelity loss**. These are explicit
design assumptions; choosing this prior for a funded run still requires validation.
The TEMF settings correspond to `alpha = 0`, `guidance_scale = 1`,
`use_temf_target = true`, unit branch weights, and no teacher loss in the
[pinned configuration](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/configs/loss/temf.yaml).
The sampling and target formulas follow the
[pinned loss implementation and inherited helpers](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/src/training/loss.py).

### Variables and direction

- `a_clean`, `a_obs`: aligned 16 kHz mono clean and degraded waveforms of the same
  underlying speech, shaped `[batch, samples]` with equal lengths within a batch.
- `c = E(a_clean)`, `y = E(a_obs)`: compressed complex STFTs, represented as
  real/imaginary channels with shape `[batch, 2, frequency, audio_frame]`.
- `h = extract_nat_context(a_obs)`: frozen context from the **degraded waveform**,
  shaped `[batch, audio_frame, 1536]` after channel concatenation and alignment.
- `z`: Gaussian prior with the same shape as `c`.
- `t`, `r`: current and destination **flow times**, with `0 <= r <= t <= 1`.
  Flow time is separate from the waveform's audio-frame axis and the optimizer step.
- `U(theta, x, y, t, r, h)`: direct prediction of **average velocity**, shaped like `x`.
  `theta` includes the restoration backbone, time MLPs, adapters, and FiLM heads.

The training path is `x_t = (1 - t)*c + t*z`, with analytic velocity `z - c`.
Inference runs backward from `t = 1` toward `t = 0`. The observed coefficients `y`
are a separate condition; they are not substituted for the Gaussian prior here.
Time vectors `t` and `r` have shape `[batch]`. `broadcast_time` reshapes them to
`[batch, 1, 1, 1]` for coefficient arithmetic; scalar loss weights stay `[batch]`.

### Frozen WavJEPA-Nat context

The mono baseline duplicates the observed waveform into both input channels, as in
the authors' mono evaluation. It adds no real binaural information. Nat supplies
representations learned with noise and reverberation; it does not output separated
speech/noise signals or clean coefficients. Channel concatenation is our explicit
FiLM design choice, not an upstream speech-restoration result.

```python
NAT_ID = "labhamlet/wavjepa-nat-base"
NAT_REVISION = "15d95ff67fa98117b17e83a1653bbca97877ff6f"
# Load the reviewed model AND processor from this revision before either loop.
# Freeze every encoder parameter; keep trainable adapters outside this helper.

@no_grad()
def extract_nat_context(a_obs):
    assert a_obs.ndim == 2                  # equal-length mono batch at 16 kHz
    dual_mono = stack((a_obs, a_obs), dim=1) # [batch, 2, samples]
    inputs = nat_processor(
        [clip.cpu().numpy() for clip in dual_mono],
        sampling_rate=16000, return_tensors="pt",
    )
    features, timestamps_ms = wavjepa_nat(inputs["input_values"].to(encoder_device))
    # Released output: [batch, 2, encoder_frame, 768], plus timestamps in ms.
    h = concat((features[:, 0], features[:, 1]), dim=-1)
    h = align_to_stft_frames(h, timestamps_ms / 1000, stft_frame_times(a_obs))
    return h.detach().clone()               # ordinary tensor for trainable FiLM
```

The executable helper is [`src/audio/nat.py`](../src/audio/nat.py). The pinned processor's
RMS normalization applies only to the encoder input; waveform scaling used by `E`
and `D` is preserved. The model performs its own segment normalization. Mock-based
tests cover channel ordering, timestamp interpolation, and inference-mode tensor
conversion; actual pretrained extraction and segment-boundary alignment still
need validation. Old mono-wavJEPA feature caches must not be reused. Details are
in the [encoder contract](wavjepa-nat.md).

### Shared audio transforms and velocity network

```python
# Pseudocode: use identical STFT, compression and waveform scaling in train/inference.
def E(waveform):
    S = STFT(waveform, config=stft_config)
    C = polar(abs(S)**0.3, angle(S))          # zero magnitude maps to zero
    return stack_channels(real(C), imag(C))

def D(coefficients, original_length):
    C = complex_from_channels(coefficients)
    S = polar(abs(C)**(1 / 0.3), angle(C))    # inverse compression, preserving phase
    return inverse_STFT(S, config=stft_config, length=original_length)

def U(theta, x, y, t, r, h):
    # Reuse SGMSE's positional helper and time MLP; supply both flow times.
    e = concat(sgmse_positional(t, nf // 2), sgmse_positional(r, nf // 2))
    return adapted_sgmse_ncsnpp(x, y, embedding=e, nat_context=h)
```

The executable backbone is [`src/models/components/network.py`](../src/models/components/network.py), backed
by pinned SGMSE NCSN++ source. It downsamples both frequency and audio time, pads
those dimensions to compatible sizes, and crops the final output back. Its output
is interpreted as mean velocity, not a diffusion score. There is no division by
noise level in this adapter.

The upstream time MLP and each residual time projection use PyTorch `RMSNorm`
(`eps=1e-4`, no affine weights). Nat FiLM follows each residual block's second
GroupNorm, with context interpolated to that block's audio-frame resolution.
This is a documented conditioning adaptation, **not an exact sCM architecture**.
Fixed positional frequencies come from SGMSE's helper; no Gaussian width-16
embedding is used. The new backbone/EMA format cannot resume the earlier lab
U-Net or production RMFSR checkpoints.

The convolution, attention, skip and FIR-resampling calculations remain upstream
code. The configuration uses only bottleneck attention and native PyTorch FIR
operators. This model is not causal, and there is no real-time latency claim.
The animation predates this backbone change and illustrates the method only.

### TEMF time sampling and schedules

```python
def sample_flow_times(batch_size, progress):
    # progress is completed optimizer updates / planned total, in [0, 1].
    boundary_fraction = 0.9 - 0.8 * sigmoid(8 * (progress - 0.6))
    n_boundary = floor(batch_size * boundary_fraction)
    n_interval = batch_size - n_boundary

    # Boundary examples train the diagonal t == r (local velocity).
    diagonal = sigmoid(normal(mean=-0.4, std=1.0, size=n_boundary))

    # Interval examples use independent draws, sorted so t >= r.
    a = sigmoid(normal(mean=-0.4, std=1.0, size=n_interval))
    b = sigmoid(normal(mean=-0.4, std=1.0, size=n_interval))
    t = concat(diagonal, maximum(a, b))
    r = concat(diagonal, minimum(a, b))
    return t, r
```

The boundary fraction decreases as training progresses, increasing the proportion
of interval examples. The released sampler does **not** explicitly insert exact
`t = 1, r = 0` examples. Its endpoints are interior in exact arithmetic; floating-point
rounding and the released near-diagonal tolerance must be handled consistently.
The velocity mixture below uses `mix = 0.3 - 0.2*progress`.

### One training update

```python
freeze_parameters(wavjepa_nat)
wavjepa_nat.eval()
theta.train()
# Optimizer owns theta only. Keep input-derivative operations deterministic.

for step in range(planned_updates):
    progress = step / planned_updates
    a_clean, a_obs = next_aligned_augmented_batch()
    c, y = E(a_clean), E(a_obs)

    h = extract_nat_context(a_obs)
    h = stop_gradient(h)                    # cache only this frozen representation

    t, r = sample_flow_times(batch_size=len(c), progress=progress)
    T, R = broadcast_time(t), broadcast_time(r)
    z = standard_normal_like(c)
    x = (1 - T) * c + T * z
    v_path = z - c

    with no_grad():
        v_diagonal = U(theta, x, y, t, t, h)
        mix = 0.3 - 0.2 * progress
        v_ref = where((T > 0) & (T < 1),
                      mix * v_path + (1 - mix) * v_diagonal, v_path)
    v_ref = stop_gradient(v_ref)

    # Continuous MeanFlow branch: differentiate inputs, not the target's parameters.
    interval = ~isclose(r, t, rtol=1e-5, atol=1e-8)
    J = zeros_like(x)
    if any(interval):
        def interval_velocity(state, current_time, destination_time):
            return U(theta, state, y[interval], current_time,
                     destination_time, h[interval])

        _, J[interval] = JVP(
            interval_velocity,
            primals=(x[interval], t[interval], r[interval]),
            tangents=(v_ref[interval], ones_like(t[interval]),
                      zeros_like(r[interval])),
        )

    target = stop_gradient(v_ref - (T - R) * J)
    prediction = U(theta, x, y, t, r, h)     # ordinary parameter-gradient graph

    delta = maximum(t - r, 0.001)           # shape [batch], not [batch, 1, 1, 1]
    weight = 1 / ((1 + delta) * sqrt(1 + delta*delta))
    error_per_example = mean_over_nonbatch_axes((prediction - target)**2)
    loss = mean_over_batch(weight * error_per_example)

    optimizer.zero_grad()
    loss.backward()
    require_finite(loss, theta.gradients)   # a failure rejects this optimizer update
    record_preclipping_gradient_norm(theta)
    if gradient_clipping_is_configured:
        clip_gradients(theta, configured_threshold)
    optimizer.step()
    update_EMA_if_configured(theta)
    record_metrics_and_checkpoint_if_due()
```

`JVP` must retain input differentiation even though its contribution to `target`
is detached before parameter backpropagation. Do not retain a higher-order parameter
graph for the target branch. The complete directional derivative is
`J = D_x U * v_ref + partial_t U`, with `r`, observed `y`, and frozen `h` held fixed.
It is not just the time partial derivative. When `r = t`, the correction vanishes;
the released continuous branch also skips it for numerically close times.

Only the fresh `prediction` branch supplies parameter gradients for this loss.
Those gradients reach the CNN, time MLPs, adapters, and FiLM heads, but not WavJEPA-Nat.
The loss compares **predicted velocity with a detached velocity target**. There is
no extra comparison of final restored audio against clean audio in this baseline,
and no coefficient-head `1/t²` conversion is needed for a direct velocity head.

The continuous branch does not apply the target clamp from TEMF's optional discrete
branch. Optimizer, learning rate, clipping, EMA, and checkpoint intervals are run
configuration choices, not prescribed by this pseudocode. The animation instead
uses one small SGD update and a fixed example to make the calculation visible.

### Sampling a restored waveform

```python
@no_grad()
def restore(a_obs, NFE, sampling_weights, seed):
    assert NFE >= 1
    theta = sampling_weights                # explicitly choose EMA or raw weights
    theta.eval()
    wavjepa_nat.eval()
    y = E(a_obs)
    h = extract_nat_context(a_obs)          # one encoder call, reused at every step
    x = standard_normal_like(y, seed=seed)  # same prior convention as training

    times = linspace(1.0, 0.0, NFE + 1)     # simple baseline inference grid
    for t, r in adjacent_pairs(times):
        velocity = U(theta, x, y, t, r, h)
        x = x - (t - r) * velocity

    return D(x, original_length=length(a_obs))
```

For **NFE = 1**, the predicted endpoint is `z - U(theta, z, y, 1, 0, h)`, followed
by inverse compression and inverse STFT. The CNN's output is the velocity to
subtract, not the clean endpoint itself. Larger NFE values repeatedly update the
state; they do not guarantee better audio. The uniform grid is an explicit baseline,
not a claim to reproduce every sampler setting used by the TEMF authors.

Sampling has **no clean target, JVP, loss, or backpropagation**. NFE counts velocity
network evaluations; WavJEPA-Nat feature extraction and STFT/inverse STFT add their own
cost. Keep waveform scaling, frame alignment, compression, and the prior consistent
between training and inference. A future streaming implementation must also maintain
encoder/model state and overlap-add boundaries across chunks.

## TEMF fidelity requirements

Reference commit: `78c531cca978a23113b262cd0b6bede2197b1916`.

- Verify the actual `TEMFLoss` class and its inherited sampling/target helpers.
- Port temporal equilibrium weighting, boundary scheduling, JVP tangents,
  stop-gradient placement, and reduction conventions with numerical parity tests.
- Record the exact source configuration; distinguish released defaults from paper ablations.
- Resolve the Gaussian versus degraded-centered prior explicitly before implementation.
- Treat compressed STFT, WavJEPA-Nat, FiLM, and any prior change as speech-specific adaptations.
- Specify guidance behavior rather than silently omitting or adding classifier-free guidance.
- Keep full MeanFlow derivatives. Do not silently remove a derivative term for stability.

This is a TEMF-based speech-restoration redesign, not a reproduction of the paper's
ImageNet architecture or evidence that its image-generation results transfer to speech.

## Initialization and evaluation

The existing 125k checkpoint predicts coefficients. Its output head and optimizer
state cannot be reinterpreted as a direct velocity predictor. Choose and document
fresh initialization or an explicit transfer strategy before launching training.

First validate flow signs/endpoints, FiLM identity behavior, feature alignment,
frozen-encoder gradients, objective parity, finite JVPs, and actual parameter count.
Short runs establish execution and stability only. A quality experiment must compare
held-out restoration, intelligibility, loudness, and compute cost at fixed NFE values.
Training duration, hardware budget, and launch are separate decisions.

## Project isolation

This is an independent repository for the 300M redesign, with its own Git history.
The existing RMFSR repository is a reference implementation, not this project's
production entry point. No training checkpoints, datasets, credentials, or cloud
configuration are included.

Existing RMFSR training and its allocated GPU remain separate. Project setup does
not launch training, create resources, or change any running process.

## References

- [TEMF source at the pinned commit](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/src/training/loss.py)
- [TEMF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Tu_Temporal_Equilibrium_MeanFlow_Bridging_the_Scale_Gap_for_One-Step_Generation_CVPR_2026_paper.pdf)
- [FiLM paper](https://arxiv.org/abs/1709.07871)
- [sCM conditioning and stabilization](https://arxiv.org/html/2410.11081v2#S4.SS1)
- [Bottleneck conditioning for speech enhancement](https://arxiv.org/abs/2606.22591)
- [WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms](https://arxiv.org/abs/2509.23238) — Yuksel et al. (2025).
- [Official WavJEPA implementation](https://github.com/labhamlet/wavjepa)
- [Pinned WavJEPA-Nat encoder](https://huggingface.co/labhamlet/wavjepa-nat-base/tree/15d95ff67fa98117b17e83a1653bbca97877ff6f)
