from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.eval import evaluate
from src.train import train
from tests.test_lightning import smoke_config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = smoke_config(tmp_path_factory.mktemp("train"))
    _, objects = train(cfg)
    return objects["trainer"].checkpoint_callback.last_model_path


def eval_config(output, checkpoint, weights):
    directory = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=directory, version_base="1.3"):
        return compose(config_name="eval", overrides=[
            "experiment=smoke", f"output_dir={output}", f"ckpt_path={checkpoint}",
            f"weights={weights}", "logger=[]",
        ])


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_evaluation_loads_selected_weights_without_optimizer(tmp_path, checkpoint, weights):
    cfg = eval_config(tmp_path, checkpoint, weights)
    metrics, objects = evaluate(cfg)
    saved = torch.load(checkpoint, weights_only=False)
    torch.testing.assert_close(objects["model"].net.state_dict(), expected_network(saved, weights))
    assert torch.isfinite(metrics["val/coefficient_mse"])
    assert objects["trainer"].global_step == 0
    assert objects["trainer"].optimizers == []
    assert not list(tmp_path.rglob("*.ckpt"))


def expected_network(saved, weights):
    state, prefix = saved["state_dict"], "net."
    if weights == "ema":
        state, prefix = saved["callbacks"]["restoration_ema"]["averaged"], "module."
    return {key.removeprefix(prefix): value for key, value in state.items()
            if key.startswith(prefix)}


def test_evaluation_requires_checkpoint_and_records_failure(tmp_path):
    cfg = eval_config(tmp_path, "null", "raw")
    with pytest.raises(ValueError, match="explicit ckpt_path"):
        evaluate(cfg)
    assert "explicit ckpt_path" in (tmp_path / "error.log").read_text()
