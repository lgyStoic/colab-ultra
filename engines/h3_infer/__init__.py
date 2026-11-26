"""MiniMax-H3-Image inference with precomputed prompt embeddings."""

from importlib import import_module

from .config import MiniMaxH3DiTConfig, MiniMaxH3PDDConfig, MiniMaxH3VAEConfig

__all__ = [
    "MiniMaxH3DiTConfig", "MiniMaxH3PDDConfig", "MiniMaxH3VAEConfig",
    "MiniMaxH3DiT", "MiniMaxH3PDDHeads", "MiniMaxH3ImageVAE",
    "MiniMaxH3Scheduler", "MiniMaxH3ImagePipeline",
]

_MODEL_EXPORTS = {
    "MiniMaxH3DiT": ".models.dit",
    "MiniMaxH3PDDHeads": ".models.dit",
    "MiniMaxH3ImageVAE": ".models.vae",
    "MiniMaxH3Scheduler": ".runtime.scheduler",
    "MiniMaxH3ImagePipeline": ".pipeline",
}


def __getattr__(name):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODEL_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
