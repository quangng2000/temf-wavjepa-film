# What has and has not been proved

## MeanFlow and reverse integration

For a differentiable trajectory `X(t)` and a differentiable average velocity
`u(t)` satisfying `(t-r) u(t) = X(t)-X(r)`, Lean derives:

```text
u(t) = v(t) - (t-r) du/dt
```

Here `r` is fixed. The separate chain-rule theorem shows that, with speech
context and `r` fixed, the derivative is the network JVP in direction `(v, 1)`
through state and current time. It is not merely a partial time derivative.
The state can be any real normed vector space; this theorem treats one output
coordinate. Displacement and differentiability are assumptions, not properties
proved for a trained network. No existence/convergence result is claimed.

For an exact average velocity, `x_r = x_t - (t-r) u` reaches the correct endpoint.
With prediction error `e`, endpoint squared error is `(t-r)² e²`.
This is also proved for the sum over any finite coefficient vector.
The oracle average velocity is not assumed equal to an arbitrary bootstrapped
JVP target. A counterexample shows that reducing error to a corrected target
can increase error to the oracle. It is not evidence that a particular run does so.

For a coefficient head `u=(x-p)/t`, equivalent weighted velocity loss is
`w/t² * (p-target)²`, for `t != 0`. The new model will predict velocity directly;
this conversion is a reference/migration check, not the proposed output head.

## TEMF release choices

Reference commit: `78c531cca978a23113b262cd0b6bede2197b1916`.

- [TEMFLoss](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/src/training/loss.py)
  starts at line 686; the earlier lines 652–679 belong to the preceding class.
- [Released configuration](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/configs/loss/temf.yaml)
  specifies the continuous branch and default schedules/weights used here.

The proved weight uses `d=max(t-r,0.001)` and
`w=1/((1+d)*sqrt(1+d²))`. It does not cover optional weight normalization or clamps.
It is positive and bounded by one, but a fixed positive weight cannot bound an
arbitrarily large target-induced gradient. No theorem says TEMF prevents spikes.

For progress in `[0,1]`, boundary proportion is
`0.9 - 0.8*sigmoid(8*(progress-0.6))`. It decreases and lies strictly between
0.1 and 0.9; it does not reach those exact endpoints during this interval.
The code's outside-range guards and integer batch split are not formalized.
Sorted logistic draws lie strictly inside `(0,1)` in real arithmetic: this
validates time ordering, not the distribution of a random-number generator.
There is no explicit exact-full-jump branch in this release's sampler.

The mix ratio lies between 0.1 and 0.3 during training. Its mixed velocity differs
from the analytic path velocity by `(1-mix)*(predicted-analytic)`. Consequently,
the pure-trajectory MeanFlow theorem does not by itself verify the entire mixed
bootstrapped training algorithm. CFG, near-diagonal tolerances, stochastic
expectations, and the full PyTorch implementation remain outside this proof suite.

## FiLM and frozen wavJEPA

FiLM is `(1+scale)*feature+shift`. Zero scale/shift gives identity.
Holding conditioning fixed along flow time multiplies the feature tangent by
`1+scale`. A bound `abs(scale)<=B` implies a tangent amplification bound `1+B`.
The bound must actually be enforced or established; identity initialization
alone does not enforce it after training. Unrestricted FiLM can amplify a tangent
arbitrarily or erase a feature. It does not know which features are important.

A separate parameter-derivative theorem shows that adapters can learn even when
encoder features are fixed. Freezing wavJEPA means not updating its parameters;
it does not freeze the adapters. Autograd behavior still needs a CPU parity test.

## Boundaries of the result

- Exact real-number mathematics, not floating-point/bfloat16 execution.
- No proof of global neural-network Jacobian bounds, optimizer stability, or convergence.
- No proof of better STOI, loudness, perceptual quality, or generalization.
- No decision between a Gaussian and degraded-centered prior; path statements
  expose the prior explicitly. A nonzero noise floor leaves noise at time zero.
- No proof that compressed-coefficient MSE equals perceptual waveform error.
- No verified STFT implementation, data alignment, normalization layer, model
  architecture, hardware performance, or justification for 300M parameters.

The purpose is to eliminate identifiable mathematical mistakes before spending
compute, while keeping the remaining empirical and implementation questions explicit.
