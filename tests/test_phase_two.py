"""Fast checks for phase two. Run from the repository root:

    python -m pytest tests/test_phase_two.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phase_two.agent import DQNAgent, LinearEpsilon
from phase_two.config import Config
from phase_two.preprocess import make_env
from phase_two.replay_buffer import ReplayBuffer

ROOT = Path(__file__).resolve().parents[1]


def test_skin_is_an_exact_copy_of_the_notebook():
    notebook = json.loads((ROOT / "pacman_dqn.ipynb").read_text())
    cells = {tag: "".join(c["source"]) for c in notebook["cells"]
             for tag in c["metadata"].get("tags", []) if tag in ("swift-sprites", "swift-stage")}
    module = (ROOT / "phase_two" / "swift_skin.py").read_text()
    copied = module.split("# ---- BEGIN NOTEBOOK COPY ----\n")[1].split("\n# ---- END NOTEBOOK COPY ----")[0]
    assert copied == cells["swift-sprites"] + "\n\n\n" + cells["swift-stage"]


def test_epsilon_schedule_is_linear_in_environment_steps():
    eps = LinearEpsilon(1.0, 0.1, 100_000)
    assert eps(0) == 1.0 and eps(100_000) == pytest.approx(0.1) and eps(10**7) == pytest.approx(0.1)
    assert eps(50_000) == pytest.approx(0.55)


def test_terminal_drops_future_value_but_truncation_keeps_it():
    agent = DQNAgent(9, Config(), torch.device("cpu"))
    last = agent.target.layers[-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.constant_(last.bias, 5.0)  # target network now predicts 5 for every action
    rewards = torch.tensor([1.0, 1.0, 1.0])
    next_obs = torch.zeros(3, 4, 84, 84, dtype=torch.uint8)
    terminated = torch.tensor([True, False, False])  # row 1 = truncated, row 2 = ordinary step
    targets = agent.targets(rewards, next_obs, terminated)
    assert targets.tolist() == pytest.approx([1.0, 1.0 + 0.99 * 5, 1.0 + 0.99 * 5])
    assert not targets.requires_grad


def _synthetic_episodes(n_episodes, rng):
    """Yield (obs, action, reward, next_obs, terminated, truncated) like FrameStackObservation would."""
    frame_id = 0
    for episode in range(n_episodes):
        length = int(rng.integers(1, 12))
        frames = []
        for _ in range(length + 1):
            frame_id += 1
            frame = np.zeros((84, 84), np.uint8)
            frame[0, 0], frame[0, 1] = frame_id % 256, frame_id // 256  # every frame unique
            frames.append(frame)
        stack = lambda t: np.stack([frames[max(0, t + k)] for k in range(-3, 1)])  # "reset" padding
        for t in range(length):
            end = t == length - 1
            yield stack(t), t % 9, float(t), stack(t + 1), end and episode % 2 == 0, end and episode % 2 == 1


def test_replay_rebuilds_real_transitions_across_episodes_and_wraparound():
    rng = np.random.default_rng(0)
    replay = ReplayBuffer(capacity=64)
    real = {}
    for obs, action, reward, next_obs, terminated, truncated in _synthetic_episodes(60, rng):
        replay.add(obs, action, reward, next_obs, terminated, truncated)
        real[obs.tobytes() + next_obs.tobytes()] = (action, min(reward, 1.0), terminated, truncated)
    assert replay.count > replay.capacity  # the ring has wrapped
    obs, actions, rewards, next_obs, terminated, truncated = replay.sample(512, rng)
    for i in range(512):
        key = obs[i].tobytes() + next_obs[i].tobytes()
        assert key in real, "sampled a stack that mixes episodes or was never observed"
        assert real[key] == (actions[i], rewards[i], terminated[i], truncated[i])


def test_reset_never_stacks_frames_from_the_previous_game():
    env = make_env(200)
    try:
        env.reset(seed=1)
        for _ in range(50):
            env.step(env.action_space.sample())
        obs, _ = env.reset(seed=2)
        assert all(np.array_equal(obs[0], frame) for frame in obs)
    finally:
        env.close()


def test_train_checkpoint_and_resume(tmp_path):
    from phase_two import train

    run_dir = tmp_path / "run"
    train.main(["--smoke", "--episodes", "2", "--run-dir", str(run_dir)])
    first = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)
    assert first["episode"] == 2 and first["replay_restored_on_resume"] is False
    train.main(["--resume", str(run_dir / "checkpoints" / "latest.pt"), "--episodes", "3"])
    with (run_dir / "training.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["episode"]) for r in rows] == [1, 2, 3]
    assert int(rows[2]["total_steps"]) > int(rows[1]["total_steps"]) == first["agent"]["steps"]
    final = torch.load(run_dir / "checkpoints" / "final.pt", weights_only=False)
    assert final["episode"] == 3
    assert any(not torch.equal(first["agent"]["online"][k], v) for k, v in final["agent"]["online"].items())
    for name in ("training_dashboard.png", "evaluation.csv", "demos/episode_0000.gif", "demos/episode_0002.gif"):
        assert (run_dir / name).exists(), name
