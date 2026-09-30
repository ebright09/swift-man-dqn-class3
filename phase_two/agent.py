"""The DQN agent: epsilon-greedy acting, an online and a target network, and one learning step.

Schedules are counted in environment steps (agent decisions), across all episodes:
- epsilon decays linearly from epsilon_start to epsilon_end over epsilon_decay_steps,
- one update happens every train_every steps once warmup_steps have been collected,
- the target network is overwritten with the online network every target_sync_steps.
Evaluation uses its own fixed epsilon and never advances any of these counters.
"""
import numpy as np
import torch
from torch import nn

from .model import DQN, build_optimizer


class LinearEpsilon:
    """epsilon(step) for a linear decay measured in environment steps."""

    def __init__(self, start, end, decay_steps):
        self.start, self.end, self.decay_steps = start, end, decay_steps

    def __call__(self, step):
        fraction = min(1.0, step / self.decay_steps) if self.decay_steps > 0 else 1.0
        return self.start + fraction * (self.end - self.start)


class DQNAgent:
    def __init__(self, n_actions, config, device):
        self.n_actions, self.config, self.device = n_actions, config, device
        self.online = DQN(n_actions).to(device)
        self.target = DQN(n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())  # start identical
        self.target.eval()
        self.optimizer = build_optimizer(self.online, config.learning_rate)
        self.epsilon = LinearEpsilon(config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
        self.steps = 0     # environment steps taken in training (drives every schedule)
        self.updates = 0   # gradient updates made

    @torch.no_grad()
    def act(self, obs, epsilon, rng):
        """Explore with probability epsilon, otherwise play the highest Q-value."""
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        screens = torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0)
        return int(self.online(screens).argmax(dim=1).item())

    @torch.no_grad()  # targets are detached: no gradient flows into the target network
    def targets(self, rewards, next_obs, terminated):
        """reward + gamma * max_a' Q_target(next, a'), with no future value after a true game over.

        Truncation (the time limit) is deliberately NOT treated as terminal: the game could
        have continued, so those transitions still bootstrap from the next screen.
        """
        future = self.target(next_obs).max(dim=1).values
        return rewards + self.config.gamma * (~terminated).float() * future

    def learn(self, batch):
        obs, actions, rewards, next_obs, terminated, truncated = [
            torch.as_tensor(x, device=self.device) for x in batch]
        # 1. The online network's value for the move that was actually taken.
        values = self.online(obs).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        # 2. The Bellman target for each transition.
        targets = self.targets(rewards, next_obs, terminated)
        # 3. Huber loss is quadratic for small errors, linear for big ones: fewer wild updates.
        loss = nn.functional.smooth_l1_loss(values, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite. Try a smaller --learning-rate.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())

    def observe_step(self, replay, rng):
        """Advance the step counter; learn and sync the target when the schedules say so."""
        self.steps += 1
        loss = None
        c = self.config
        if self.steps >= c.warmup_steps and replay.count >= c.warmup_steps and self.steps % c.train_every == 0:
            loss = self.learn(replay.sample(c.batch_size, rng))
        if self.steps % c.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss

    # --- checkpoint support ---
    def state_dict(self):
        cpu = lambda sd: {k: v.detach().cpu().clone() for k, v in sd.items()}
        return {"online": cpu(self.online.state_dict()), "target": cpu(self.target.state_dict()),
                "optimizer": self.optimizer.state_dict(), "steps": self.steps, "updates": self.updates}

    def load_state_dict(self, state):
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps, self.updates = state["steps"], state["updates"]
