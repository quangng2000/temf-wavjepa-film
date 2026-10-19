# Augmentation and data-loading dependencies

Installed as packages, not copied into this repository:

- [audiomentations 0.43.1](https://github.com/iver56/audiomentations): MIT.
  Recorded-noise SNR mixing, filters, clipping, tanh, masks, gain and MP3 wrapper.
- [pyroomacoustics 0.10.0](https://github.com/LCAV/pyroomacoustics): MIT.
  Image-source simulation supplies matched direct/full RIRs from one room scene.
- [fast-mp3-augment 0.2.0](https://pypi.org/project/fast-mp3-augment/): LGPL-2.1-or-later.
  Used through audiomentations with delay compensation; no local codec rewrite.
- [SciPy](https://scipy.org/): BSD-3-Clause. `signal.fftconvolve` applies RIRs.
- [TorchData 0.11.0](https://github.com/meta-pytorch/data): BSD-3-Clause.
  StatefulDataLoader provides iteration snapshots and worker-state restoration.
- [PyTorch 2.8.0](https://pytorch.org/): `nn.utils.get_total_norm` calculates the
  pre-clipping gradient norm and rejects nonfinite totals. Lightning still clips.

Local code coordinates clean/observed branches, common timing/gain, manifests and
RNG state. Audiomentations ApplyImpulseResponse is deliberately not used because
its output peak normalization and uncompensated propagation delay do not implement
our paired-target contract. Library room simulation + convolution avoid rewriting DSP.
See [the recipe and limits](../docs/augmentation.md); no paper-parity claim is made.
