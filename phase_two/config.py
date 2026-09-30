"""Every tunable number in one place, with its unit.

Units used below:
- "environment step" = one agent decision = 4 emulator frames (frame skip is 4).
- "episode" = one game, ending at game over or the per-game step cap.
- "update" = one gradient step on one minibatch.
"""
import argparse
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # --- training budget ---
    episodes: int = 500                 # episodes (games) to train, total across resumes
    max_steps_per_episode: int = 3000   # environment steps; about 200 s of game time
    seed: int = 42

    # --- exploration (epsilon-greedy) ---
    epsilon_start: float = 1.0          # probability of a random move at step 0
    epsilon_end: float = 0.1            # floor, reached after epsilon_decay_steps
    epsilon_decay_steps: int = 100_000  # environment steps for the linear decay

    # --- learning ---
    gamma: float = 0.99                 # discount per environment step
    learning_rate: float = 1e-4         # Adam step size
    batch_size: int = 32                # transitions per update
    train_every: int = 4                # environment steps between updates
    warmup_steps: int = 10_000          # environment steps collected before the first update
    target_sync_steps: int = 10_000     # environment steps between target-network copies
    replay_gb: float = 0.0              # replay memory budget in GiB; 0 = auto (40% of free RAM, max 2)
    reward_clip: float = 1.0            # learning rewards are clipped to [-1, 1]; scores stay raw

    # --- evaluation and recording ---
    eval_every: int = 25                # episodes between evaluation demonstrations
    eval_epsilon: float = 0.05          # fixed random-move probability during evaluation
    eval_seeds: list = field(default_factory=lambda: [101, 202, 303, 404, 505])
    checkpoint_every: int = 100         # episodes between numbered checkpoints
    gif_decisions: int = 300            # environment steps recorded per GIF (20 s of game time)
    gif_stride: int = 4                 # keep every 4th recorded step -> at most 75 GIF frames
    gif_frame_ms: int = 67              # display time per GIF frame -> 4x playback speed
    swift_mode: bool = True             # SWIFT-MAN skin on GIFs and the live window
    live_window: bool = True            # show evaluation demos live (falls back to GIF only)

    def to_dict(self):
        return asdict(self)


SMOKE = dict(  # a tiny run that touches every code path in about a minute
    episodes=4, max_steps_per_episode=300, epsilon_decay_steps=1_000, warmup_steps=200,
    target_sync_steps=250, eval_every=2, checkpoint_every=2, replay_gb=0.05,
    eval_seeds=[101, 202], gif_decisions=120, live_window=False,
)


def parse_args(argv=None, description="Train a DQN on Ms. Pac-Man."):
    parser = argparse.ArgumentParser(description=description)
    for f in fields(Config):
        default = Config().__getattribute__(f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif isinstance(default, list):
            parser.add_argument(flag, type=int, nargs="+", default=None)
        else:
            parser.add_argument(flag, type=type(default), default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run for checking setup")
    parser.add_argument("--resume", help="path to a training checkpoint (e.g. runs/<run>/checkpoints/latest.pt)")
    parser.add_argument("--run-dir", help="output folder (default: runs/<timestamp>)")
    args = parser.parse_args(argv)
    overrides = {f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None}
    base = {**SMOKE} if args.smoke else {}
    return Config(**{**base, **overrides}), args, overrides
