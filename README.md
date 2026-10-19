# TEMF + wavJEPA FiLM restoration

Status: project scope and Lean proof suite established; redesigned model is not implemented or trained.

## Animation

[![End-to-end preview: waveform representations, FiLM conditioning, and waveform reconstruction](docs/media/film-temf-scm-preview.gif)](docs/media/film-temf-scm-motion.mp4)

**[Watch the full end-to-end animation — 2 minutes 28 seconds](docs/media/film-temf-scm-motion.mp4)** ·
[Download MP4](https://raw.githubusercontent.com/quangng2000/temf-wavjepa-film/main/docs/media/film-temf-scm-motion.mp4)

The animation follows one synthetic recording from waveform to waveform:

1. **Represent the damaged audio:** complex STFT coefficients and a separate frozen
   wavJEPA context branch.
2. **Construct the training pair:** transform the aligned clean waveform and build
   the intermediate flow state.
3. **Condition the CNN:** positional time embeddings, sCM-inspired double
   normalization, and FiLM scaling and shifting from the frozen context.
4. **Train with velocity predictions:** form the full JVP target, apply TEMF temporal
   weighting, and update the trainable parameters.
5. **Run inference:** sample with the observed audio and context, without a clean target.
6. **Return to a waveform:** inverse magnitude compression, inverse FFT, windowed
   overlap-add, and the output waveform.

A gold cell follows the same location; blue and coral show positive and negative
values. The GIF previews the input branches, FiLM, and waveform reconstruction.

The STFT, JVP, SGD update, sampling states, and inverse STFT are computed by a small
toy model using synthetic audio. A fixed waveform projection stands in for the
frozen wavJEPA encoder; it is not run here. This is a process explanation, not an
audio-quality result. The Gaussian prior and conditioning placement illustrate
design choices still to be finalized. See the
[reproducible source, checks and limitations](docs/animation/README.md).

## Mathematics first

All formal proofs live in [`lean/`](lean/README.md), with pinned Lean/Mathlib versions
and an axiom-auditing checker. They cover the MeanFlow identity, reverse solver,
TEMF weights/schedules, and FiLM derivatives. See [scope](lean/SCOPE.md) for the
assumptions and limits. CPU implementation parity comes next, before GPU experiments.

## Agreed scope

- Target approximately **300 million trainable restoration parameters**.
- Keep the existing **89,280,768-parameter wavJEPA encoder frozen**.
- Use wavJEPA features to condition restoration through trainable **FiLM** adapters.
- Predict the **mean velocity directly**, rather than converting a clean-coefficient prediction into velocity.
- Implement and test TEMF's objective and time sampling against the pinned authors' code.

The combined inference model would contain approximately 389 million parameters.
The exact count depends on the backbone and adapter design, which remain to be specified.

## Proposed signal path

1. Encode degraded audio as compressed complex STFT coefficients.
2. Independently extract frozen wavJEPA features from the same degraded waveform.
3. Align those features to audio-frame positions and project them through trainable adapters.
4. Generate FiLM scale and shift for selected restoration blocks.
5. Condition the restoration backbone on the flow state, degraded coefficients,
   current/destination flow times, and the aligned wavJEPA context.
6. Predict mean velocity and integrate it to the clean endpoint.
7. Decompress coefficients and reconstruct the waveform with inverse STFT.

FiLM will start as an identity modulation: `(1 + scale) * features + shift`,
with zero-initialized final scale/shift projections. Placement and normalization
must be explicit experimental choices; they are not established quality improvements.

## TEMF fidelity requirements

Reference commit: `78c531cca978a23113b262cd0b6bede2197b1916`.

- Verify the actual `TEMFLoss` class and its inherited sampling/target helpers.
- Port temporal equilibrium weighting, boundary scheduling, JVP tangents,
  stop-gradient placement, and reduction conventions with numerical parity tests.
- Record the exact source configuration; distinguish released defaults from paper ablations.
- Resolve the Gaussian versus degraded-centered prior explicitly before implementation.
- Treat compressed STFT, wavJEPA, FiLM, and any prior change as speech-specific adaptations.
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
- [Pinned wavJEPA encoder](https://huggingface.co/labhamlet/wavjepa-base/tree/6be4a5093f6c9adbbd55e00b6e5b8f067aa03345)
