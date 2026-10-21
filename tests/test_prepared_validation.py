"""Invalid prepared batches cannot silently change the effective-batch objective."""
from dataclasses import replace

import pytest
import torch

from src.flow.objective import TEMFObjective
from src.flow.prepared import prepare_batch
from tests.test_prepared import make_batch
from tests.test_target import AnalyticVelocity


@pytest.mark.parametrize("field", ["noise", "temporal_weight"])
def test_wrong_prepared_batch_length_is_rejected(field):
    objective, batch = TEMFObjective(), make_batch()
    prepared = prepare_batch(objective.upstream, batch[0][:, None], 0)
    with pytest.raises(ValueError):
        invalid = replace(prepared, **{field: getattr(prepared, field)[:2]})
        objective(AnalyticVelocity(), batch, 0, prepared=invalid)


def test_preparation_rejects_noncontinuous_recipe():
    objective, batch = TEMFObjective(), make_batch()
    objective.upstream.cfg.alpha.initial_value = .5
    objective.upstream.cfg.alpha.end_value = .5
    with pytest.raises(ValueError):
        prepare_batch(objective.upstream, batch[0][:, None], 0)


def test_preparation_does_not_modify_the_clean_reference():
    objective, batch = TEMFObjective(), make_batch()
    reference = batch[0][:, None]
    original = reference.clone()
    prepare_batch(objective.upstream, reference, 0)
    torch.testing.assert_close(reference, original, rtol=0, atol=0)
