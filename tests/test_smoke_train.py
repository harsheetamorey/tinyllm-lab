"""The smoke script's resume tolerance policy."""

from __future__ import annotations

import torch

from scripts.smoke_train import CPU_PARAM_TOLERANCE, param_tolerance


def test_cpu_must_resume_almost_exactly_while_gpu_gets_one_learning_rate():
    assert param_tolerance(torch.device("cpu"), 6e-4) == CPU_PARAM_TOLERANCE
    assert param_tolerance(torch.device("mps"), 6e-4) == 6e-4
    assert param_tolerance(torch.device("cuda"), 1e-3) == 1e-3
