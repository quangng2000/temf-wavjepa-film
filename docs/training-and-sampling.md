# Training and sampling

[Project](../README.md) · [Setup and commands](training.md) · [Sources and adaptations](../third_party/README.md)

This document describes the implemented **TEMF + WavJEPA-Nat** speech-restoration
method. A frozen audio encoder conditions a trainable mean-velocity network.
Training uses aligned clean and degraded speech; inference needs only degraded
speech and a Gaussian initial state.

**Implementation status.** The scaffold is CPU smoke-tested. Full-model training,
GPU performance, and restoration quality have not been established. The algorithms
below describe the implementation, not a validated experimental result.

## Contents

- [Notation and signal path](#notation-and-signal-path)
- [Audio representation and conditioning](#audio-representation-and-conditioning)
- [Flow path and time sampling](#flow-path-and-time-sampling)
- [TEMF training objective](#temf-training-objective)
- [Algorithm 1 · One training update](#one-training-update)
- [Algorithm 2 · Speech restoration](#sampling-a-restored-waveform)
- [Implementation scope and verification](#implementation-scope-and-verification)
- [References](#references)

<a id="agreed-scope"></a>
<a id="implemented-signal-path"></a>

## Notation and signal path

- $a_{\mathrm{clean}}, a_{\mathrm{obs}}$: aligned clean and degraded waveforms at 16 kHz.
- $c=\mathcal{E}(a_{\mathrm{clean}})$ and $y=\mathcal{E}(a_{\mathrm{obs}})$: compressed
  complex STFT coefficients, stored as real and imaginary channels.
- $h$: frozen WavJEPA-Nat features from the degraded waveform, aligned to STFT frames.
- $z\sim\mathcal{N}(0,I)$: a unit Gaussian prior with the same shape as $c$.
- $t,r$: current and destination **flow times**, satisfying $0\le r\le t\le1$.
- $u_\theta(x,t,r\mid y,h)$: the network's direct prediction of **mean velocity**.
- $\operatorname{sg}(\cdot)$: stop-gradient; its argument is constant during parameter backpropagation.
- $B$: batch size; $D$: number of real coefficient entries per example.
- $s$: completed optimizer updates; $S$: planned total; $p=s/S$: training progress.

Flow time is separate from audio time and the optimizer-update counter. Coefficient
batches have shape $B\times2\times F\times T$; context has shape
$B\times T\times1536$, where $F$ and $T$ index frequency bins and audio frames.

The trainable parameters $\theta$ include SGMSE NCSN++, its time controls, and Nat
FiLM adapters: **325,497,126 parameters**, excluding the frozen encoder. The Nat
inference parameter count must be measured separately from its full checkpoint,
which also includes pretraining modules.

<a id="training-and-sampling-pseudocode"></a>

## Audio representation and conditioning

### Compressed complex STFT

Let $S_a=\operatorname{STFT}(a)$. With compression exponent $\gamma=0.3$, define

$$
\begin{aligned}
C_a &= |S_a|^\gamma\exp\!\bigl(\mathrm{i}\arg S_a\bigr),\\
\mathcal{E}(a) &= \begin{bmatrix}\operatorname{Re}C_a\\ \operatorname{Im}C_a\end{bmatrix},\\
\mathcal{D}(c;L) &= \operatorname{iSTFT}\!\left(
  |C|^{1/\gamma}\exp\!\bigl(\mathrm{i}\arg C\bigr);\,L\right),
  \qquad C=c_{\mathrm{Re}}+\mathrm{i}c_{\mathrm{Im}}.
\end{aligned}
\tag{1}
$$

Zero magnitude maps to zero. Training and inference share the same transform:
512-point FFT, 160-sample hop, Hann window, centered constant padding, and normalized
STFT. Reconstruction uses the original waveform length $L$.
[Implementation: `Spectrum`](../src/audio/spectrum.py).

### Frozen WavJEPA-Nat context

The mono waveform is duplicated into the encoder's two input channels. Channel
features are concatenated, then interpolated using the encoder's released timestamps:

$$
\begin{aligned}
(H^{(1)},H^{(2)},\tau)
  &= \Phi_{\mathrm{Nat}}\!\left(P_{\mathrm{Nat}}([a_{\mathrm{obs}},a_{\mathrm{obs}}])\right),\\
h &= \operatorname{sg}\!\left(
  \operatorname{Align}_{\mathrm{STFT}}([H^{(1)};H^{(2)}],\tau)\right).
\end{aligned}
\tag{2}
$$

Here $P_{\mathrm{Nat}}$ is the pinned input processor and $\Phi_{\mathrm{Nat}}$ is
the frozen encoder. Its input normalization does not change the waveform used by
$\mathcal{E}$. Dual-mono input adds no spatial information. Nat provides context;
it does not emit separated speech/noise signals or clean coefficients.

The model and processor use revision `15d95ff67fa98117b17e83a1653bbca97877ff6f` of
[`labhamlet/wavjepa-nat-base`](https://huggingface.co/labhamlet/wavjepa-nat-base/tree/15d95ff67fa98117b17e83a1653bbca97877ff6f).
Channel concatenation is our conditioning design. Pretrained extraction and segment
boundaries still need real-data validation. [Encoder contract](wavjepa-nat.md) ·
[Implementation: `FrozenNat`](../src/audio/nat.py).

### Velocity network and FiLM

Both flow times enter the upstream SGMSE positional embedding and time MLP. Nat
features modulate each residual block after its second GroupNorm:

$$
\begin{aligned}
e(t,r) &= [\operatorname{PE}(t);\operatorname{PE}(r)],\\
u_\theta(x,t,r\mid y,h)
  &= \operatorname{NCSN{+}{+}}_\theta(x,y;e(t,r),h),\\
\operatorname{FiLM}_\ell(g,h)
  &= \bigl(1+\Delta\gamma_\ell(h)\bigr)\odot g+\beta_\ell(h).
\end{aligned}
\tag{3}
$$

FiLM heads start at zero, making the initial modulation an identity. Context is
interpolated to each block's audio-frame resolution. The time-MLP output and
residual time projections use affine-free RMSNorm with $\epsilon=10^{-4}$.
This is a documented adaptation, **not an exact sCM architecture**.

The backbone uses SGMSE's fixed positional frequencies, convolution, bottleneck
attention, skip connections, and native PyTorch FIR operations. It downsamples
both frequency and audio time, with padding and final cropping. It is **noncausal**.
Its output is mean velocity, with no diffusion-score or noise-level rescaling.
[Network](../src/models/components/network.py) · [FiLM](../src/models/components/film.py).

## Flow path and time sampling

### Gaussian-to-speech path

Training interpolates between clean coefficients and Gaussian noise:

$$
\boxed{x_t=(1-t)c+tz,\qquad v_{\mathrm{path}}=z-c.}
\tag{4}
$$

At $t=0$ the path is clean; at $t=1$ it is Gaussian noise. The degraded coefficients
$y$ remain a separate condition. Inference integrates in the reverse direction,
from $t=1$ to $t=0$.

### Progress-dependent time pairs

Let $\sigma(q)=(1+e^{-q})^{-1}$ and let $\mathcal{Q}$ denote the distribution
of $\sigma(\xi)$ for $\xi\sim\mathcal{N}(-0.4,1)$. The released boundary schedule is

$$
\rho(p)=0.9-0.8\,\sigma\!\bigl(8(p-0.6)\bigr),
\qquad B_{\mathrm{diag}}=\lfloor B\rho(p)\rfloor.
\tag{5}
$$

Draw the batch's time pairs as follows; each draw is independent:

$$
(t_i,r_i)=
\begin{cases}
(q_i,q_i),\quad q_i\sim\mathcal{Q}, & i\le B_{\mathrm{diag}},\\[3pt]
(\max(q_i,q'_i),\min(q_i,q'_i)),\quad q_i,q'_i\sim\mathcal{Q},
  & i>B_{\mathrm{diag}}.
\end{cases}
\tag{6}
$$

Diagonal examples learn local velocity. As training progresses, their allocation
decreases and interval examples become more common. The integer allocation uses
flooring, so small batches need not realize the nominal fraction exactly. The
sampler does **not** explicitly insert the full jump $(t,r)=(1,0)$; logit-normal
draws are interior in exact arithmetic.

## TEMF training objective

The equations below specialize the pinned implementation to its configured
continuous branch: `alpha=0`, unit branch weights, unit guidance scale, no condition
dropout, and no teacher loss. There is no auxiliary speech-fidelity loss.

### Reference velocity

With the released mixture schedule $m(p)=0.3-0.2p$, define

$$
\bar v=\operatorname{sg}\!\left[
\begin{cases}
m(p)\,v_{\mathrm{path}}+\bigl(1-m(p)\bigr)u_\theta(x_t,t,t\mid y,h),
  & 0<t<1,\\[3pt]
v_{\mathrm{path}}, & t\in\{0,1\}.
\end{cases}\right]
\tag{7}
$$

The diagonal prediction participates in the numerical target but receives no
parameter gradient through that target.

### Full directional derivative and detached target

For interval examples, differentiate along the state and current-time direction,
holding destination time and conditioning fixed:

$$
\begin{aligned}
J_\theta
  &= \operatorname{JVP}_{(x,t,r)}[u_\theta]\!\left[(\bar v,1,0)\right]\\
  &= D_xu_\theta(x_t,t,r\mid y,h)\,\bar v
     +\partial_tu_\theta(x_t,t,r\mid y,h),\\[3pt]
U^\star &= \operatorname{sg}\!\left[\bar v-(t-r)J_\theta\right].
\end{aligned}
\tag{8}
$$

The JVP is the **complete state-plus-time derivative**, not just the time partial.
The implementation sets $J_\theta=0$ when `torch.isclose(r, t)` is true, using
`rtol=1e-5` and `atol=1e-8`. At exact equality, the correction vanishes. Input
differentiation is retained for the JVP, but no higher-order parameter graph is
retained through $U^\star$. The optional discrete-branch target clamp is not used.

### Weighted velocity error

For each example, floor the interval only when computing its temporal weight:

$$
\delta_i=\max(t_i-r_i,10^{-3}),
\qquad w_i=\frac{1}{(1+\delta_i)\sqrt{1+\delta_i^2}}.
\tag{9}
$$

The scalar batch objective is

$$
\boxed{
\mathcal{L}(\theta)=\frac{1}{B}\sum_{i=1}^{B}
 w_i\,\frac{\left\|u_\theta(x_{t_i},t_i,r_i\mid y_i,h_i)-U_i^\star\right\|_2^2}{D}.
}
\tag{10}
$$

Reduction averages all real/imaginary coefficient entries, then averages across
examples. Weights are not normalized by their batch mean. Parameter gradients
flow only through the prediction outside stop-gradient: the backbone, time
controls, and FiLM adapters learn; WavJEPA-Nat remains frozen. A direct velocity
head needs no coefficient-head $1/t^2$ conversion.

### One training update

**Algorithm 1 · Continuous TEMF update**

**Given:** an aligned batch, current parameters $\theta$, optimizer state, and progress $p=s/S$.
**Return:** updated restoration parameters and EMA; the Nat encoder stays fixed.

$$
\begin{array}{r l}
1. & (c,y)\gets\bigl(\mathcal{E}(a_{\mathrm{clean}}),\mathcal{E}(a_{\mathrm{obs}})\bigr)\\[4pt]
2. & h\gets\operatorname{FrozenNatContext}(a_{\mathrm{obs}})\\[4pt]
3. & (t,r)\gets\operatorname{SampleTimes}(B,p)\quad\text{using (5)--(6)}\\[4pt]
4. & z\sim\mathcal{N}(0,I),\quad x_t\gets(1-t)c+tz,\quad v_{\mathrm{path}}\gets z-c\\[4pt]
5. & \bar v\gets\operatorname{ReferenceVelocity}(\theta,x_t,t,y,h,p)\quad\text{using (7)}\\[4pt]
6. & U^\star\gets\operatorname{sg}\!\left[\bar v-(t-r)J_\theta\right]\quad\text{using (8)}\\[4pt]
7. & \mathcal{L}\gets\operatorname{WeightedVelocityError}(\theta,U^\star)\quad\text{using (9)--(10)}\\[4pt]
8. & g\gets\nabla_\theta\mathcal{L}\quad\text{after clearing accumulated gradients}\\[4pt]
9. & \text{Reject the update if the loss, diagnostics, or gradients are nonfinite.}\\[4pt]
10. & \text{Record }\|g\|_2;\quad g\gets\operatorname{ClipNorm}(g,1)\\[4pt]
11. & \theta\gets\operatorname{AdamWStep}(\theta,g)\quad\text{with optimizer state}\\[4pt]
12. & \theta_{\mathrm{EMA}}\gets0.999\,\theta_{\mathrm{EMA}}+0.001\,\theta
\end{array}
$$

The algorithm shows current default optimizer/clipping/EMA choices, which remain
configurable. Input derivatives are deterministic and computed in FP32. EMA is
initialized from the starting network. Checkpoints are saved every 1,000 updates;
validation runs every 5,000. These are run settings, not requirements of TEMF.

[Training hooks](../src/models/temf_module.py) · [Objective adapter](../src/flow/objective.py) ·
[EMA](../src/callbacks/ema.py) · [Stability checks](../src/callbacks/stability.py).

### Sampling a restored waveform

**Algorithm 2 · Reverse mean-flow sampling**

**Given:** degraded audio $a_{\mathrm{obs}}$, explicit raw or EMA weights $\hat\theta$,
number of evaluations $K\ge1$, and a random seed. **Return:** restored waveform $\hat a$.

$$
\begin{array}{r l}
1. & y\gets\mathcal{E}(a_{\mathrm{obs}}),\quad h\gets\operatorname{FrozenNatContext}(a_{\mathrm{obs}})\\[4pt]
2. & x^{(0)}\gets z,\quad z\sim\mathcal{N}(0,I)\quad\text{using the supplied seed}\\[4pt]
3. & t_k\gets1-k/K,\quad k=0,\ldots,K\\[4pt]
4. & \textbf{for }k=0,\ldots,K-1:\\[4pt]
5. & \quad v^{(k)}\gets u_{\hat\theta}(x^{(k)},t_k,t_{k+1}\mid y,h)\\[4pt]
6. & \quad x^{(k+1)}\gets x^{(k)}-(t_k-t_{k+1})v^{(k)}\\[4pt]
7. & \textbf{return }\hat a=\mathcal{D}(x^{(K)};\operatorname{length}(a_{\mathrm{obs}}))
\end{array}
$$

For one evaluation, the complete restoration is

$$
\boxed{\hat a=\mathcal{D}\!\left(z-u_{\hat\theta}(z,1,0\mid y,h);\,L\right).}
\tag{11}
$$

The CNN predicts the velocity to subtract, not the endpoint itself. Sampling uses
no clean target, JVP, loss, or backpropagation. Nat context is computed once and
reused across all $K$ evaluations. STFT and inverse STFT add their own cost.

The uniform inference grid is our baseline, not every TEMF sampler setting.
More evaluations do not guarantee better speech. Waveform scaling, compression,
feature alignment, and the prior must match training. A future streaming model
would also need explicit state and chunk-boundary handling.
[Solver](../src/flow/solver.py) · [Audio restoration entrypoint](../src/sample.py).

<a id="mathematics-first"></a>
<a id="temf-fidelity-requirements"></a>
<a id="initialization-and-evaluation"></a>

## Implementation scope and verification

**Upstream fidelity.** The trainer executes pinned TEMF method bodies through an
audio adapter. Numerical tests compare loss, parameter gradients, targets, weights,
and schedules against an independent reference. This is a speech adaptation, not
TEMF's ImageNet architecture or evidence that its image results transfer to speech.

**Formal checks.** The [Lean proofs](../lean/README.md), with pinned Lean/Mathlib
versions and axiom auditing, cover the MeanFlow identity, reverse solver, schedules,
weights, and FiLM derivatives. [Proof scope](../lean/SCOPE.md) states the assumptions;
formal identities do not establish learned restoration quality.

**Initialization.** Restoration weights start from scratch; the pretrained Nat
encoder is frozen. Earlier coefficient-prediction RMFSR checkpoints, including the
125k snapshot, cannot be reinterpreted as this direct velocity head. Old lab U-Net
and EMA formats also require an explicit migration.

**Evaluation.** Tests cover flow signs/endpoints, FiLM identity, alignment, frozen
encoder behavior, objective parity, JVPs, and parameter count. Mock encoder tests
and synthetic smoke runs are not real-speech validation. Quality must be measured
on held-out audio using restoration error, intelligibility, loudness, and compute
cost at fixed evaluation budgets.

**Isolation.** This repository does not launch or control the existing RMFSR
training process or GPU. It contains no datasets, checkpoints, credentials, or cloud
configuration. The [animation](animation/README.md) is a small illustrative
calculation that predates the backbone change, not this full training run.

## References

- [Pinned TEMF implementation](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/src/training/loss.py) · [Pinned configuration](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/configs/loss/temf.yaml)
- [Temporal Equilibrium MeanFlow](https://openaccess.thecvf.com/content/CVPR2026/papers/Tu_Temporal_Equilibrium_MeanFlow_Bridging_the_Scale_Gap_for_One-Step_Generation_CVPR_2026_paper.pdf)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [sCM conditioning and stabilization](https://arxiv.org/html/2410.11081v2#S4.SS1)
- [Bottleneck conditioning for speech enhancement](https://arxiv.org/abs/2606.22591)
- [WavJEPA](https://arxiv.org/abs/2509.23238) · [Official implementation](https://github.com/labhamlet/wavjepa)
