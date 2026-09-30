"""Evaluate saved agents on five fixed games, and compare them fairly.

    python -m phase_two.play --checkpoint runs/<run>/checkpoints/final.pt

Three policies are compared under identical settings (environment, seeds, evaluation
epsilon, step cap), each clearly labeled:
- "random actions":    ignores the screen entirely,
- "untrained network": the network before any learning (random weights, 5% random moves),
- "trained network":   the checkpoint you pass in.
Every game's raw score is printed, not only the best one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .model import DQN
from .preprocess import make_env

RANDOM_LABEL, UNTRAINED_LABEL, TRAINED_LABEL = "random actions", "untrained network", "trained network"


def load_network(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DQN(checkpoint["n_actions"]).to(device)
    model.load_state_dict(checkpoint["agent"]["online"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(model, n_actions, config, device, gif_path=None, record_best=False, live=None, label=""):
    """Play each evaluation seed once. model=None means the random-action baseline.

    Uses a fresh environment of its own, never touches replay memory, and never changes
    weights or the training step counter. Scores are raw game points (no clipping).
    """
    from PIL import Image as PILImage
    from .swift_skin import SwiftSkin

    env = make_env(config.max_steps_per_episode)
    scores, lengths, capped = [], [], []
    best, best_frames = -float("inf"), []
    try:
        for index, seed in enumerate(config.eval_seeds):
            rng = np.random.default_rng(seed + 10_000)
            obs, _ = env.reset(seed=seed)
            capture = gif_path is not None and (record_best or index == 0)
            watch = live is not None and index == 0 and live.open(f"SWIFT-MAN · {label}")
            skin = SwiftSkin(seed) if capture and config.swift_mode else None
            total, frames = 0.0, []
            for step in range(config.max_steps_per_episode):
                if capture and step < config.gif_decisions and step % config.gif_stride == 0:
                    screen = env.render()
                    frame = skin.render(screen, total) if skin else PILImage.fromarray(screen).copy()
                    frames.append(frame)
                    if watch:
                        live.show(frame, f"score {total:.0f}")
                if model is None or rng.random() < config.eval_epsilon:
                    action = int(rng.integers(n_actions))
                else:
                    screens = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
                    action = int(model(screens).argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            if watch:
                live.finish()
            scores.append(float(total))
            lengths.append(step + 1)
            capped.append(bool(truncated))
            if capture and (not record_best or total > best):
                best, best_frames = total, frames
    finally:
        env.close()
    if gif_path is not None and best_frames:
        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        best_frames[0].save(gif_path, save_all=True, append_images=best_frames[1:],
                            duration=config.gif_frame_ms, loop=0)
    return {"label": label, "seeds": list(config.eval_seeds), "scores": scores,
            "mean": float(np.mean(scores)), "steps": lengths, "time_limited": capped,
            "eval_epsilon": 1.0 if model is None else config.eval_epsilon}


def print_table(results):
    width = max(len(r["label"]) for r in results) + 2
    print(f"{'Seed':>6} " + "".join(f"{r['label']:>{width}}" for r in results))
    for i, seed in enumerate(results[0]["seeds"]):
        print(f"{seed:>6} " + "".join(f"{r['scores'][i]:>{width}.0f}" for r in results))
    print(f"{'Mean':>6} " + "".join(f"{r['mean']:>{width}.1f}" for r in results))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint (.pt)")
    parser.add_argument("--baseline", help="untrained checkpoint (default: untrained.pt next to --checkpoint)")
    parser.add_argument("--out", help="folder for comparison.json and demos/ (default: the checkpoint's run folder)")
    parser.add_argument("--no-live-window", action="store_true")
    args = parser.parse_args(argv)

    from .device import select_device
    from .live_window import LiveWindow

    checkpoint_path = Path(args.checkpoint)
    baseline_path = Path(args.baseline) if args.baseline else checkpoint_path.parent / "untrained.pt"
    run_dir = Path(args.out) if args.out else checkpoint_path.parent.parent
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config(**saved["config"])
    device = select_device(saved["n_actions"], config)
    trained, _ = load_network(checkpoint_path, device)
    untrained, _ = load_network(baseline_path, device)
    live = LiveWindow(config.live_window and not args.no_live_window, config.gif_frame_ms)

    print(f"Evaluating on seeds {config.eval_seeds}, epsilon {config.eval_epsilon}, "
          f"step cap {config.max_steps_per_episode} (same settings for every policy)...")
    results = [evaluate(None, saved["n_actions"], config, device, label=RANDOM_LABEL),
               evaluate(untrained, saved["n_actions"], config, device, label=UNTRAINED_LABEL),
               evaluate(trained, saved["n_actions"], config, device, run_dir / "demos" / "final_best.gif",
                        record_best=True, live=live, label=TRAINED_LABEL)]
    live.close()
    print_table(results)
    print(f"Change vs untrained network: {results[2]['mean'] - results[1]['mean']:+.1f} | "
          f"vs random actions: {results[2]['mean'] - results[0]['mean']:+.1f}")
    n = len(config.eval_seeds)
    print(f"final_best.gif shows the best of the {n} trained games; judge the agent by all {n} scores.")
    comparison = {"checkpoint": str(checkpoint_path), "episode": saved["episode"],
                  "environment_steps": saved["agent"]["steps"], "max_steps_per_episode": config.max_steps_per_episode,
                  "results": results}
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print("Saved", run_dir / "comparison.json", "and", run_dir / "demos" / "final_best.gif")
    return comparison


if __name__ == "__main__":
    main()
