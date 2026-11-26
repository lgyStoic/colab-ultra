"""Distilled velocity Euler schedule for MiniMax-H3-Image."""

import torch

from ..config import MiniMaxH3PDDConfig


class MiniMaxH3Scheduler:
    """Six indexed Euler steps and the terminal sigma.

    The DiT predicts data-ward velocity: x0 = x_t + sigma * v_data.
    Timesteps use DiffSynth's float32 sigma -> time -> sigma conversion.
    """

    def __init__(self, config: MiniMaxH3PDDConfig | None = None):
        self.config = config if config is not None else MiniMaxH3PDDConfig()
        self.num_inference_steps = self.config.num_inference_steps
        release_sigmas = torch.tensor(self.config.sigmas, dtype=torch.float32, device="cpu")
        self.sigmas = 1.0 - (1.0 - release_sigmas)
        self.timesteps = self.sigmas[:-1] * 1000
        self.model_timesteps = 1.0 - self.timesteps / 1000

    def _sigma_at(self, step_index):
        if not 0 <= step_index < self.num_inference_steps:
            raise IndexError(f"Step index {step_index} is outside the PDD schedule")
        return self.sigmas[step_index]

    def predict_x0(self, v_data, step_index, sample):
        return sample + self._sigma_at(step_index) * v_data

    def step(self, v_data, step_index, sample):
        """Advance x_t by (sigma_t - sigma_next) * v_data."""
        sigma = self._sigma_at(step_index)
        sigma_next = self.sigmas[step_index + 1]
        # Keep DiffSynth's operation order with its noise-ward velocity.
        return sample + (-v_data) * (sigma_next - sigma)
