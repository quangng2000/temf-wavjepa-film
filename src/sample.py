"""Restore a mono WAV from a trusted lab checkpoint; explicit raw/EMA selection."""
# Root discovery must precede project imports for the direct-script entry point.
# ruff: noqa: E402

import json
from pathlib import Path

import hydra
import rootutils
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.pairs import read_audio
from src.flow.solver import solve
from src.utils import extras


def load_model(path, weights, device):
    # Lightning checkpoints include Python state: accept only your own trusted files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("lab_format") != "sgmse-temf-v2":
        raise ValueError("Old lab checkpoint format requires explicit migration")
    model = instantiate(OmegaConf.create(checkpoint["run_config"]["model"]))
    model.run_config = checkpoint["run_config"]
    model.on_load_checkpoint(checkpoint)
    model.conditioner.load(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if weights == "ema":
        averaged = checkpoint["callbacks"]["restoration_ema"]["averaged"]
        shadow = {key.removeprefix("module."): value for key, value in averaged.items()
                  if key.startswith("module.")}
        model.net.load_state_dict(shadow, strict=True)
    elif weights != "raw":
        raise ValueError("weights must be raw or ema")
    return model.to(device).eval(), checkpoint["global_step"]


@torch.no_grad()
def restore(model, waveform, nfe, seed):
    observed = model.spectrum.encode(waveform)
    positions = model.spectrum.frame_times(observed.shape[-1], waveform.device)
    context = model.conditioner(waveform, positions)
    coefficients = solve(model.net, (observed, context), nfe, seed)
    restored = model.spectrum.decode(coefficients, waveform.shape[-1])
    if not torch.isfinite(restored).all():
        raise FloatingPointError("Nonfinite decoded waveform")
    return restored


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample")
def main(cfg):
    extras(cfg)
    output = Path(cfg.output)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Choose a new output path")
    model, step = load_model(cfg.checkpoint, cfg.weights, cfg.device)
    waveform = read_audio(cfg.input).unsqueeze(0).to(cfg.device)
    restored = restore(model, waveform, cfg.nfe, cfg.seed).cpu()[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, restored.numpy(), 16000, subtype="FLOAT")
    report_path.write_text(json.dumps(dict(checkpoint=cfg.checkpoint, step=step,
        weights=cfg.weights, nfe=cfg.nfe, seed=cfg.seed, peak=restored.abs().max().item(),
        rms=restored.square().mean().sqrt().item(), gain_applied_db=0), indent=2))


if __name__ == "__main__":
    main()
