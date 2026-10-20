"""Sample globally, encode small groups, and accumulate the full-batch mean gradient."""
import torch

from src.flow.prepared import prepare_batch


def sample_effective_batch(model, batch, step):
    clean = batch["clean"]
    config = model.spectrum.recipe
    shape = (len(clean), 1, 2, config["n_fft"] // 2 + 1, clean.shape[-1] // config["hop"] + 1)
    return prepare_batch(model.objective.upstream, clean.new_empty(shape), step)


def backward_microbatch(model, batch, prepared, backward):
    encoded = model.encode_batch(batch)
    if not all(torch.isfinite(value).all() for value in encoded):
        raise FloatingPointError("Nonfinite encoded microbatch; optimizer update rejected")
    loss, metrics = model.objective(model.net, encoded, prepared.step, prepared)
    if not all(torch.isfinite(value).all() for value in (loss, *metrics.values())):
        raise FloatingPointError("Nonfinite TEMF loss/target/JVP; optimizer update rejected")
    fraction = len(batch["clean"]) / model.effective_batch_size
    backward(loss * fraction)
    return dict(metrics, loss=loss.detach())


def accumulate_gradients(model, batch, step, backward):
    if len(batch["clean"]) != model.effective_batch_size:
        raise ValueError("Loader batch must equal the configured effective batch size")
    prepared = sample_effective_batch(model, batch, step)
    totals = {}
    for start in range(0, len(batch["clean"]), model.microbatch_size):
        selection = slice(start, start + model.microbatch_size)
        chunk = {key: value[selection] for key, value in batch.items()}
        metrics = backward_microbatch(model, chunk, prepared.slice(selection), backward)
        fraction = len(chunk["clean"]) / model.effective_batch_size
        merge_metrics(totals, metrics, fraction)
    return {key: value.sqrt() if key.endswith("_rms") else value
            for key, value in totals.items()}


def merge_metrics(totals, metrics, fraction):
    for key, value in metrics.items():
        value = value.detach()
        contribution = value.square() if key.endswith("_rms") else value
        totals[key] = totals.get(key, 0) + contribution * fraction
