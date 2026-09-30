"""Train the DQN on Ms. Pac-Man, with evaluation demos, checkpoints, and honest resume.

    python -m phase_two.train --smoke                  # about a minute; checks every code path
    python -m phase_two.train                          # the full default run (500 episodes)
    python -m phase_two.train --resume runs/<run>/checkpoints/latest.pt

Ctrl+C saves a resumable checkpoint, the metrics, and the plot, then prints the resume command.
"""
import csv
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent
from .config import Config, parse_args
from .device import select_device
from .live_window import LiveWindow
from .play import UNTRAINED_LABEL, evaluate
from .plot import plot_run
from .preprocess import WRAPPER_SETTINGS, make_env
from .replay_buffer import ReplayBuffer, capacity_for_budget

TRAIN_FIELDS = ["episode", "score", "steps", "total_steps", "epsilon", "mean_loss", "updates",
                "elapsed_seconds", "terminated", "truncated"]
EVAL_FIELDS = ["episode", "total_steps", "mean_score", "min_score", "max_score"]
PACKAGES = ["torch", "gymnasium", "ale-py", "numpy", "opencv-python-headless", "Pillow", "matplotlib"]


class Run:
    """Everything that must survive a resume, in one place."""

    def __init__(self, config, run_dir, n_actions, device):
        self.config, self.run_dir, self.n_actions, self.device = config, run_dir, n_actions, device
        self.agent = DQNAgent(n_actions, config, device)
        self.act_rng = np.random.default_rng(config.seed)          # exploration coin flips
        self.replay_rng = np.random.default_rng(config.seed + 1)   # minibatch sampling
        self.episode = 0                                           # completed training episodes
        self.elapsed_before = 0.0                                  # seconds from earlier sessions
        self.started = time.monotonic()
        self.history, self.evals = [], []

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started

    # --- checkpoints ---
    def save(self, path, note=""):
        payload = {
            "kind": "phase_two_training_checkpoint", "note": note,
            "agent": self.agent.state_dict(), "n_actions": self.n_actions, "config": self.config.to_dict(),
            "episode": self.episode, "elapsed_seconds": self.elapsed(),
            "rng": {"act": self.act_rng.bit_generator.state, "replay": self.replay_rng.bit_generator.state,
                    "torch": torch.get_rng_state(), "python": random.getstate()},
            "replay_restored_on_resume": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, checkpoint):
        self.agent.load_state_dict(checkpoint["agent"])
        self.act_rng.bit_generator.state = checkpoint["rng"]["act"]
        self.replay_rng.bit_generator.state = checkpoint["rng"]["replay"]
        torch.set_rng_state(checkpoint["rng"]["torch"])
        random.setstate(checkpoint["rng"]["python"])
        self.episode = checkpoint["episode"]
        self.elapsed_before = checkpoint["elapsed_seconds"]
        self.history = [row for row in read_rows(self.run_dir / "training.csv") if int(row["episode"]) <= self.episode]
        self.evals = [row for row in read_rows(self.run_dir / "evaluation.csv") if int(row["episode"]) <= self.episode]

    # --- logs ---
    def write_logs(self):
        write_rows(self.run_dir / "training.csv", TRAIN_FIELDS, self.history)
        write_rows(self.run_dir / "evaluation.csv", EVAL_FIELDS, self.evals)
        plot_run(self.run_dir)


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preflight(config, n_actions, device, run_dir):
    """Check every moving part on throwaway objects before spending real training time."""
    from PIL import Image
    from .swift_skin import SwiftSkin

    env = make_env(config.max_steps_per_episode)
    try:
        obs, _ = env.reset(seed=config.seed)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8, obs.shape
        assert all(np.array_equal(obs[0], f) for f in obs), "reset stack should be 4 copies of one frame"
        print("  reset: 4 x 84 x 84 uint8 stack of the first frame")
        screen = env.render()
        assert screen.shape == (210, 160, 3)
        frame = SwiftSkin(0).render(screen, 0.0) if config.swift_mode else Image.fromarray(screen)
        print("  frame capture: RGB", screen.shape, "-> GIF frame", frame.size)
        probe = DQNAgent(n_actions, config, device)
        rng = np.random.default_rng(0)
        replay = ReplayBuffer(1000, config.reward_clip)
        stacks = [obs]
        for _ in range(40):
            action = probe.act(stacks[-1], 0.5, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(stacks[-1], action, reward, next_obs, terminated, truncated)
            stacks.append(next_obs)
        print("  action selection + replay insertion: 40 steps stored")
        # Global id k holds the newest frame of the k-th observation, so its stack must equal stacks[k].
        assert all(np.array_equal(replay._stacks(np.array([k]))[0], stacks[k]) for k in range(replay.count))
        print("  replay reconstruction: rebuilt stacks match the real observations")
        loss = probe.learn(replay.sample(config.batch_size, rng))
        assert np.isfinite(loss)
        print(f"  training batch: loss {loss:.4f}")
        path = run_dir / "checkpoints" / "preflight.pt"
        torch.save(probe.state_dict(), path)
        copy = DQNAgent(n_actions, config, device)
        copy.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in
                   zip(probe.online.state_dict().values(), copy.online.state_dict().values()))
        path.unlink()
        print("  checkpoint round-trip: identical weights after reload")
        gif = run_dir / "demos" / "preflight.gif"
        frame.save(gif, save_all=True, append_images=[frame, frame], duration=config.gif_frame_ms)
        with Image.open(gif) as check:
            assert check.n_frames >= 1
        gif.unlink()
        print("  GIF output: writes and reopens")
    finally:
        env.close()


def train_one_episode(run, env, replay):
    config, agent = run.config, run.agent
    episode = run.episode + 1
    obs, _ = env.reset(seed=config.seed + episode)
    score, losses, epsilon = 0.0, [], agent.epsilon(agent.steps)
    for step in range(config.max_steps_per_episode):
        epsilon = agent.epsilon(agent.steps)
        action = agent.act(obs, epsilon, run.act_rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, terminated, truncated)
        loss = agent.observe_step(replay, run.replay_rng)
        if loss is not None:
            losses.append(loss)
        obs, score = next_obs, score + reward
        if agent.steps % 5000 == 0:
            print(f"    {agent.steps:,} steps | {agent.updates:,} updates | epsilon {epsilon:.3f} | "
                  f"{run.elapsed() / 60:.1f} min", flush=True)
        if terminated or truncated:
            break
    return {"episode": episode, "score": score, "steps": step + 1, "total_steps": agent.steps,
            "epsilon": round(epsilon, 5), "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "updates": agent.updates, "elapsed_seconds": round(run.elapsed(), 2),
            "terminated": bool(terminated), "truncated": bool(truncated)}


def run_evaluation(run, live, label):
    """Separate evaluation: its own env, fixed epsilon, no replay, no counter changes."""
    was_training = run.agent.online.training
    run.agent.online.eval()
    try:
        gif = run.run_dir / "demos" / f"episode_{run.episode:04d}.gif"
        result = evaluate(run.agent.online, run.n_actions, run.config, run.device, gif, live=live, label=label)
    finally:
        run.agent.online.train(was_training)
    run.evals.append({"episode": run.episode, "total_steps": run.agent.steps, "mean_score": result["mean"],
                      "min_score": min(result["scores"]), "max_score": max(result["scores"])})
    detail = run.run_dir / "evaluation.json"
    saved = json.loads(detail.read_text()) if detail.exists() else []
    saved = [r for r in saved if r["episode"] < run.episode] + [{"episode": run.episode, **result}]
    detail.write_text(json.dumps(saved, indent=2))
    print(f"  Evaluation after {run.episode} episodes: {result['scores']} | mean {result['mean']:.1f} | GIF {gif.name}")
    return result


def main(argv=None):
    config, args, overrides = parse_args(argv)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = Config(**{**checkpoint["config"], **overrides})  # e.g. --episodes 800 extends a run
        run_dir = Path(args.resume).resolve().parent.parent
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / (("smoke_" if args.smoke else "") + stamp)
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "demos").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = make_env(config.max_steps_per_episode)
    n_actions = int(env.action_space.n)  # read from the environment, not hard-coded (9 for Ms. Pac-Man)
    print("Actions:", env.unwrapped.get_action_meanings())
    device = select_device(n_actions, config)
    run = Run(config, run_dir, n_actions, device)
    replay = ReplayBuffer(capacity_for_budget(config.replay_gb), config.reward_clip)
    live = LiveWindow(config.live_window, config.gif_frame_ms)

    info = {**config.to_dict(), "wrappers": WRAPPER_SETTINGS, "device": str(device),
            "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "packages": {p: version(p) for p in PACKAGES}, "replay_capacity": replay.capacity,
            "command": " ".join(sys.argv)}
    if checkpoint:
        run.restore(checkpoint)
        print(f"Resumed {run_dir} at episode {run.episode}, {run.agent.steps:,} environment steps, "
              f"epsilon {run.agent.epsilon(run.agent.steps):.3f}. Replay memory is NOT restored: "
              f"the agent re-collects {config.warmup_steps:,} steps before updating again.")
        info["resumed_from"] = str(args.resume)
        (run_dir / f"config_resume_{run.episode:04d}.json").write_text(json.dumps(info, indent=2, default=str))
        if run.episode % config.eval_every == 0 and all(int(r["episode"]) != run.episode for r in run.evals):
            print("The interrupted session missed this episode's evaluation; running it now.")
            run_evaluation(run, live, f"after {run.episode} episodes")
    else:
        (run_dir / "config.json").write_text(json.dumps(info, indent=2, default=str))
        print("Preflight checks:")
        preflight(config, n_actions, device, run_dir)
        run.save(run_dir / "checkpoints" / "untrained.pt", "before any learning")
        print("Baseline: evaluating the untrained network...")
        run_evaluation(run, live, UNTRAINED_LABEL)

    status = "completed"
    try:
        while run.episode < config.episodes:
            row = train_one_episode(run, env, replay)
            run.episode = row["episode"]
            run.history.append(row)
            recent = np.mean([float(r["score"]) for r in run.history[-25:]])
            print(f"Episode {run.episode}/{config.episodes} | score {row['score']:.0f} | last-25 mean {recent:.1f} | "
                  f"epsilon {row['epsilon']:.3f} | loss {row['mean_loss']:.4f} | steps {row['total_steps']:,} | "
                  f"{row['elapsed_seconds'] / 60:.1f} min", flush=True)
            write_rows(run_dir / "training.csv", TRAIN_FIELDS, run.history)
            if run.episode % config.eval_every == 0:
                run_evaluation(run, live, f"after {run.episode} episodes")
                run.write_logs()
            if run.episode % config.checkpoint_every == 0:
                run.save(run_dir / "checkpoints" / f"episode_{run.episode:04d}.pt")
                run.save(run_dir / "checkpoints" / "latest.pt")
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nCtrl+C: saving a resumable checkpoint...")
    except Exception:
        status = "failed"
        raise
    finally:
        env.close()
        live.close()
        run.save(run_dir / "checkpoints" / "latest.pt", status)
        if status == "completed":
            shutil.copyfile(run_dir / "checkpoints" / "latest.pt", run_dir / "checkpoints" / "final.pt")
        run.write_logs()
        summary = {"status": status, "completed_episodes": run.episode, "environment_steps": run.agent.steps,
                   "learning_updates": run.agent.updates, "elapsed_seconds": round(run.elapsed(), 1),
                   "elapsed_includes": "training, evaluation demos, and GIF recording, across resumes",
                   "device": str(device), "replay_capacity": replay.capacity}
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        latest = run_dir / "checkpoints" / "latest.pt"
        if status == "interrupted":
            print(f"Resume with:\n  python -m phase_two.train --resume \"{latest}\"")
        else:
            print(f"Compare against the baselines with:\n  python -m phase_two.play --checkpoint \"{run_dir / 'checkpoints' / 'final.pt'}\"")
    return run_dir


if __name__ == "__main__":
    main()
