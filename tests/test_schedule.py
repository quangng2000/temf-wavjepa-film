import json
from pathlib import Path

import pytest
import torch

from src.flow.schedule import sample_times, schedule, temporal_weight
from src.flow.target import corrected_target
from tests.test_target import AnalyticVelocity

REFERENCE = json.loads((Path(__file__).parent / "reference/temf.json").read_text())


def test_weights_match_pinned_upstream_numeric_reference():
    current = torch.tensor(REFERENCE["deltas"], dtype=torch.float64)
    expected = torch.tensor(REFERENCE["weights"], dtype=torch.float64)
    torch.testing.assert_close(temporal_weight(current, torch.zeros_like(current)),
                               expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("index", range(len(REFERENCE["steps"])))
def test_schedules_match_pinned_upstream_numeric_reference(index):
    boundary, mix = schedule(REFERENCE["steps"][index], REFERENCE["total"])
    assert boundary == pytest.approx(REFERENCE["boundary"][index], abs=1e-14)
    assert mix == pytest.approx(REFERENCE["mix"][index], abs=1e-14)


def test_full_continuous_target_matches_upstream_with_audio_tensor_layout():
    data = REFERENCE["target"]
    state = torch.tensor(data["state"], dtype=torch.float64)
    times = tuple(torch.tensor(data[key], dtype=torch.float64)
                  for key in ("current", "destination"))
    reference = torch.full_like(state, data["reference"])
    actual, _, _ = corrected_target(AnalyticVelocity(), state, times, (state, state), reference)
    expected = torch.tensor(data["expected"], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_sampler_uses_floor_boundary_count_and_sorted_independent_draws():
    reference = torch.zeros(8, 2, 2, 2)
    first = sample_times(reference, .75, torch.Generator().manual_seed(99))
    second = sample_times(reference, .75, torch.Generator().manual_seed(99))
    torch.testing.assert_close(first, second)
    assert (first[0][:6] == first[1][:6]).all()
    assert (first[0][6:] > first[1][6:]).all()
    assert (first[0] <= 1).all() and (first[1] >= 0).all()
