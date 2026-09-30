"""Pick CUDA, Apple Silicon MPS, or CPU, and prove the choice works with a real training batch."""
import platform

import numpy as np
import torch


def describe_machine():
    print(f"OS: {platform.platform()} | Python {platform.python_version()} | torch {torch.__version__}")


def select_device(n_actions, config):
    """Try each available device with one real minibatch update; fall back if any op fails."""
    from .agent import DQNAgent  # local import: agent imports model, which is cheap

    describe_machine()
    candidates = []
    if torch.cuda.is_available():
        candidates.append("cuda")
    if torch.backends.mps.is_available():
        candidates.append("mps")
    candidates.append("cpu")
    for name in candidates:
        try:
            probe = DQNAgent(n_actions, config, torch.device(name))
            screens = np.zeros((config.batch_size, 4, 84, 84), np.uint8)
            batch = (screens, np.zeros(config.batch_size, np.int64), np.ones(config.batch_size, np.float32),
                     screens, np.zeros(config.batch_size, bool), np.zeros(config.batch_size, bool))
            probe.learn(batch)  # forward, Huber loss, backward, Adam step: the whole path
            print(f"Device: {name.upper()} (training-batch smoke test passed)")
            return torch.device(name)
        except RuntimeError as error:
            if name == "cpu":
                raise
            print(f"Device FALLBACK: {name.upper()} failed the training-batch test ({error}). Trying the next one.")


if __name__ == "__main__":
    from .config import Config
    select_device(9, Config())
