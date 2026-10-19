# Mathematical checks before training

This directory contains the Lean specifications, proofs, and verification tools
for the proposed TEMF + wavJEPA FiLM restoration model.
The model is not implemented yet: these are mathematical contracts for that work,
not a verified implementation or a proof of training convergence.

## Reproduce

Install Lean's `elan` toolchain manager, then run from this directory:

```sh
lake exe cache get
python3 check.py
```

Lean 4.30.0 and Mathlib commit
`c5ea00351c28e24afc9f0f84379aa41082b1188f` are pinned.
The first command retrieves compiled Mathlib dependencies; it does not use a GPU.
The checker builds the library with warnings treated as errors and audits every
named theorem. It rejects proof holes and any axioms outside the standard
`propext`, `Classical.choice`, and `Quot.sound` foundations.
The audit and source hashes are written under ignored `.lake/`.
The committed [verification record](verification.json) records the checked
41-theorem snapshot. Re-run the checker after changes; old hashes do not certify
new source files.

## Proof map

- [Path](Temf/Path.lean): affine endpoints, analytic velocity, reverse direction,
  and the nonzero-noise-floor endpoint.
- [MeanFlow](Temf/MeanFlow.lean): differentiate endpoint displacement to derive
  the MeanFlow identity; prove the state/time chain rule for its JVP.
- [Solver](Temf/Solver.lean): exact average velocity reaches the endpoint;
  velocity error maps to endpoint error; coefficient conversion needs `1/t²`.
- [Weight](Temf/Weight.lean): the released default TEMF weight is positive,
  at most one, and cannot change the zero-error solution for a fixed target.
- [Schedule](Temf/Schedule.lean): decreasing boundary proportion, valid sorted
  time pairs, strict interior logistic samples, and tangent-mixing bounds.
- [Film](Temf/Film.lean): identity initialization, flow-time derivatives,
  conditional derivative bounds, and trainable adapters with fixed features.
- [Objective](Temf/Objective.lean): detached-target gradients and counterexamples
  to unconditional stability or endpoint-quality guarantees.

See [scope and assumptions](SCOPE.md) before interpreting a theorem as a guarantee.

## Required order of work

1. Keep these mathematical contracts checked as the design evolves.
2. Implement small CPU parity tests against the pinned TEMF source: target/JVP,
   sampler, reduction, stop-gradient, loss, and parameter gradients.
3. Check the actual neural implementation: frozen encoder, FiLM initialization,
   finite derivatives, numerical precision, and coefficient/waveform transforms.
4. Only then propose a separately authorized small stability experiment.

A passing proof suite does not authorize an experiment or establish that a 300M
model is necessary. Compute budget and training launch remain separate decisions.
