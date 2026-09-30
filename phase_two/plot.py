"""Draw training_dashboard.png from a run folder's logs.

    python -m phase_two.plot runs/<run>

Training scores (with exploration, from training games) and evaluation scores (fixed 5%
exploration, separate games) are plotted in separate panels. All scores are raw game
points; the reward clipping to [-1, 1] only affects what the network learns from.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    numeric = lambda v: v not in ("True", "False", "")
    return [{k: float(v) for k, v in row.items() if numeric(v)} for row in rows]


def plot_run(run_dir):
    run_dir = Path(run_dir)
    train = read_csv(run_dir / "training.csv")
    evals = read_csv(run_dir / "evaluation.csv")
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (a_train, a_eval), (a_loss, a_eps) = axes
    if train:
        ep = [r["episode"] for r in train]
        score = [r["score"] for r in train]
        avg = [np.mean(score[max(0, i - 24):i + 1]) for i in range(len(score))]
        a_train.plot(ep, score, alpha=0.3, label="training game (raw score)")
        a_train.plot(ep, avg, label="last-25 mean")
        a_train.legend(fontsize=8)
        a_loss.plot(ep, [r["mean_loss"] for r in train])
        a_eps.plot(ep, [r["epsilon"] for r in train])
    if evals:
        ep = [r["episode"] for r in evals]
        a_eval.plot(ep, [r["mean_score"] for r in evals], marker="o", label="mean of evaluation games")
        a_eval.fill_between(ep, [r["min_score"] for r in evals], [r["max_score"] for r in evals],
                            alpha=0.2, label="min-max")
        a_eval.legend(fontsize=8)
    eps_eval = config.get("eval_epsilon", 0.05)
    titles = ["Training score (with exploration)", f"Evaluation score (epsilon {eps_eval}, fixed seeds)",
              "Mean Huber loss per episode (clipped rewards)", "Training epsilon at episode end"]
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel="completed episode")
        ax.grid(alpha=0.2)
    a_eps.set_ylim(0, 1.05)
    fig.suptitle("Scores are raw game points. Learning rewards were clipped to [-1, 1]. "
                 "Lower loss does not mean higher scores.", fontsize=10)
    fig.tight_layout()
    path = run_dir / "training_dashboard.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot a phase-two training run.")
    parser.add_argument("run_dir")
    print("Saved", plot_run(parser.parse_args(argv).run_dir))


if __name__ == "__main__":
    main()
