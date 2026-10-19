# Project structure and configuration

[Back to the project](../README.md) · [Training commands](training.md)

The project follows the
[Lightning-Hydra-Template structure](https://github.com/ashleve/lightning-hydra-template/tree/bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144),
with speech restoration in place of the template's example task. Model equations
and sampling are documented separately in [Training and sampling](training-and-sampling.md).

## Where things live

- [src/train.py](../src/train.py): construct the configured objects and train or resume.
- [src/eval.py](../src/eval.py): evaluate checkpoint weights on a validation manifest.
- [src/sample.py](../src/sample.py): restore a recording and write a WAV file.
- [src/models/temf_module.py](../src/models/temf_module.py): Lightning training and
  validation hooks; [model components](../src/models/components) hold the backbone and FiLM.
- [src/data](../src/data): paired, augmented and synthetic LightningDataModules;
  [data components](../src/data/components) hold datasets, manifests and replay support.
- [src/utils](../src/utils): shared configuration, object construction and task utilities.
- [configs](../configs): training, evaluation and sampling configs, plus composable groups.
- [tests](../tests): numerical checks, data contracts and CPU smoke/resume coverage.
- [third_party](../third_party/README.md): pinned sources, licenses and adaptations.

The `.project-root` marker identifies the repository. Entrypoints discover that
root and load an optional `.env`; [.env.example](../.env.example) shows the format.

## Entrypoints

From the repository, either module or script syntax works:

```sh
python -m src.train experiment=smoke
python src/train.py experiment=smoke
python -m src.eval experiment=smoke ckpt_path=/absolute/checkpoints/last.ckpt
python src/eval.py experiment=smoke ckpt_path=/absolute/checkpoints/last.ckpt
```

Training uses `configs/train.yaml`; evaluation uses `configs/eval.yaml`. Evaluation
runs the validation loader supplied by the data config, without inventing a train
or test split. [See the data requirements and checkpoint options](training.md).

## Compose a run

Select a config group with `group=name`, then change individual values with dotted
overrides. Experiments bundle overrides for repeatable runs.

```sh
python -m src.train experiment=smoke trainer=cpu logger=csv
python -m src.train experiment=smoke model.optimizer.lr=1e-4
python -m src.train experiment=smoke --cfg job
```

The last command prints the composed config without running training.

- `data`, `model` and `trainer` choose the dataset, restoration model and execution settings.
- `callbacks` and `logger` contain named callback/logger configs.
- `experiment` bundles a complete experiment; `debug` applies development overrides.
- `paths` and `hydra` control directories and Hydra's run behavior.
- `extras` controls optional task utilities.
- `local` holds machine-specific overrides; `hparams_search` holds sweep settings.

Logger values live under their name, for example `logger.csv.name=trial`.
Use `logger=[]` to disable logging. Debug presets are `default`, `fast_dev_run`
and `overfit`. They shorten execution; they do not replace real data with fixtures.

```sh
python -m src.train experiment=smoke debug=fast_dev_run 'logger=[]'
python -m src.train -m experiment=smoke hparams_search=lr
```

The sweep runs two learning rates through Hydra's built-in BasicSweeper.
It records `val/coefficient_mse` as the optimization metric; it is not an adaptive search.
For private defaults, create the ignored `configs/local/default.yaml` with
`# @package _global_` on its first line. Hydra loads it when present.

The `cpu` and `gpu` trainer presets both use FP32 and one device. GPU performance,
mixed precision and distributed training have not been validated for this JVP objective.
The [training guide](training.md) explains the supported smoke checks and remaining limits.

## Run outputs and tooling

Runs go to `logs/<task_name>/runs/<date_time>/`, with checkpoints in `checkpoints/`,
CSV metrics in `metrics/version_*/`, and Hydra's config snapshot in `.hydra/`.
Sweeps use `logs/<task_name>/multiruns/<date_time>/<job_number>/`.
Set `paths.log_dir=/absolute/logs` to relocate the run tree. The existing
`output_dir=/absolute/run` override still controls checkpoint and CSV destinations.

The [Makefile](../Makefile) provides `train`, `smoke`, `eval`, `test`, `lint`,
`format` and `upstream` targets. For example, `make smoke ARGS="trainer=cpu"`;
set `PYTHON=/path/to/python` to choose an interpreter. CPU CI runs lint and tests
after fetching the pinned TEMF source. [.pre-commit-config.yaml](../.pre-commit-config.yaml)
provides optional local formatting and lint hooks.

## Compatibility

The source layout changes preserve old module paths as compatibility imports.
Current-format checkpoints and their saved Hydra targets remain loadable. Earlier
lab U-Net/EMA checkpoints and RMFSR checkpoints still require an explicit migration;
moving files does not make those architectures compatible.
