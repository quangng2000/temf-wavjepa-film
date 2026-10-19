# Animation source

The 148-second animation follows one synthetic recording through waveform encoding,
convolution, positional time conditioning, FiLM, a TEMF-style velocity target, one
SGD step, four reverse sampling steps, and waveform reconstruction. It is a numerical
explanation, not a trained
speech-restoration result or an implementation of the proposed 300M model.

## What is computed

`toy_trace.py` evaluates a small deterministic PyTorch model and saves `trace.json`.
The renderer reads those values; it does not invent an improved endpoint.

- The synthetic waveform has 5,888 samples at 16 kHz. Its actual complex STFT uses
  a 512-sample Hann window, hop 256, centered constant padding, and normalized FFT.
  Magnitude compression uses power 0.3; real and imaginary parts become channels.
  Calculation uses all 257 frequency bins and 24 frames. Animation grids display
  the first 12 bins, generally the real plane of coefficients or feature channel 1.
- Frozen context comes from a fixed random projection of aligned damaged-waveform windows.
  It stands in for JEPA features. No wavJEPA weights are loaded.
- The toy FiLM head pools that context over time and produces channel controls.
  A real restoration model still needs the planned frame-aligned adapter design.
- Two flow times use 32 fixed angular frequencies each, from 1 to 0.0001.
  Per-frame grouped feature normalization and normalized time scale/shift
  illustrate the conditioning design; this is not a full sCM/TrigFlow model.
- The direct velocity predictor uses a Gaussian-to-clean path. Its JVP direction
  is 20% analytic path velocity plus 80% predicted diagonal velocity, illustrating
  the selected TEMF mixture at 50% schedule progress. The displayed boundary
  proportion is evaluated there; this single example does not run a scheduler.
- The target is detached before weighted velocity MSE. There is no auxiliary
  speech-fidelity loss. One SGD step changes the CNN, time MLP and FiLM parameters.
- The loss comparison holds that target fixed. It does not establish convergence,
  perceptual improvement, or a decrease with a newly computed moving target.
- The sampler receives only the updated model, observed coefficients, context and prior.
  Every state is computed from its predecessor and predicted velocity; it never
  receives a clean target. Four evaluations are shown, not one-step quality.
- Reconstruction uses both predicted complex planes, inverse magnitude compression,
  and an actual inverse STFT. Three neighboring windowed inverse-FFT frames are
  illustrated; all 24 participate in reconstruction. Explicit normalized overlap-add
  is checked against the inverse STFT. No clean waveform is substituted at the end.

Blue and coral distinguish positive and negative values. Gold follows bin 6,
frame 11; after convolution this means the corresponding feature location, not
an unchanged coefficient. Smooth interpolation reveals each operation; it does
not represent extra model evaluations. Error and gradient display multipliers
are labeled. Other grids share the same magnitude-to-opacity mapping.

## Reproduce

Requirements: Python with PyTorch; Node.js with `@napi-rs/canvas`; FFmpeg with
libx264. Run from the repository root with these dependencies available:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B docs/animation/toy_trace.py
node docs/animation/render.cjs --previews
node docs/animation/render.cjs
```

`FFMPEG_BINARY` optionally selects FFmpeg. `ANIMATION_OUTPUT` optionally changes
the output folder; otherwise files go to ignored `outputs/film-temf-scm-v3/`.
`NODE_PATH` can point to an existing installation of the canvas dependency.
The renderer streams frames to FFmpeg rather than saving a full image sequence.

The trace generator checks the JVP against central finite differences, its
state/time decomposition, finite sampling states, and each numerical state update.
The saved trace has maximum JVP/finite-difference error below `2e-9`; the clean
waveform transform round-trip error is below `3e-16`, and explicit overlap-add
agrees with the inverse STFT to the precision stored in the trace.
These checks verify this toy computation, not equivalence of a future production
implementation to every TEMF or sCM component.

The media output is 1440 × 1080, 24 fps, H.264/yuv420p, with no audio track.
The README GIF is a 12-second montage: input representations (seconds 2–6),
FiLM (62–66), and reconstruction (142–146). Its cuts are previews, not solver steps.
