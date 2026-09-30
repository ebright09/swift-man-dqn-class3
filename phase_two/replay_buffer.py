"""Experience replay that stores every screen once.

Interface: add(obs, action, reward, next_obs, terminated, truncated) and sample(batch_size).

Why not store the stacks? Each observation is 4 frames of 84x84 bytes = 28,224 bytes;
storing both obs and next_obs is 56,448 bytes per transition, so 300,000 transitions
would need about 16 GiB. Consecutive stacks overlap by 3 frames, so instead we keep a
ring of single frames (7,056 bytes each) and rebuild both stacks when sampling.

Layout: slot i holds one frame plus the action/reward/flags of the step that *produced*
that frame. When a new episode starts we first store its reset frame as a "start" slot,
which is not a transition by itself. Rebuilding a stack never reaches past that start
slot; it repeats the start frame instead, matching FrameStackObservation's "reset"
padding, so frames from different episodes are never stacked together.
"""
import os
import subprocess
import sys

import numpy as np

FRAME_BYTES = 84 * 84                   # one uint8 grayscale frame
BYTES_PER_TRANSITION = FRAME_BYTES + 1 + 4 + 1 + 1 + 8  # frame, action, reward, 2 flags, start id


def available_ram_bytes():
    """Best-effort free + reclaimable RAM; falls back to total RAM."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
            page = int(out.split("page size of ")[1].split(" ")[0])
            pages = {k.strip(): int(v.strip(" .\n")) for k, v in
                     (line.split(":") for line in out.splitlines()[1:] if ":" in line)}
            return page * sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def capacity_for_budget(replay_gb):
    """Turn a memory budget into a capacity, printing the arithmetic before allocating."""
    free = available_ram_bytes()
    if replay_gb <= 0:  # auto: leave most free memory to the OS, the model, and the emulator
        replay_gb = round(min(2.0, 0.4 * free / 2**30), 2)
        print(f"Replay: auto budget {replay_gb:g} GiB (40% of free RAM, capped at 2 GiB).")
    budget = int(replay_gb * 2**30)
    capacity = budget // BYTES_PER_TRANSITION
    print(f"Replay: {BYTES_PER_TRANSITION:,} bytes per transition (one 84x84 frame + bookkeeping) "
          f"-> {capacity:,} transitions in {replay_gb:g} GiB. "
          f"(Storing both 4-frame stacks would fit only {budget // (2 * 4 * FRAME_BYTES):,}.)")
    print(f"Replay: about {free / 2**30:.1f} GiB of RAM currently free or reclaimable.")
    if budget > 0.5 * free:
        print("Replay WARNING: the budget is over half of free RAM. Consider a smaller --replay-gb.")
    if capacity < 1000:
        raise ValueError("Replay budget is too small for useful training (under 1,000 transitions).")
    return capacity


class ReplayBuffer:
    def __init__(self, capacity, reward_clip=1.0):
        self.capacity = capacity
        self.reward_clip = reward_clip
        # np.zeros memory is committed by the OS gradually, as slots are written.
        self.frames = np.zeros((capacity, 84, 84), np.uint8)
        self.actions = np.zeros(capacity, np.uint8)
        self.rewards = np.zeros(capacity, np.float32)       # clipped rewards, used for learning
        self.terminated = np.zeros(capacity, bool)
        self.truncated = np.zeros(capacity, bool)
        self.start_id = np.zeros(capacity, np.int64)        # global id of this episode's start slot
        self.count = 0                                      # slots ever written (global id of next slot)
        self._episode_start = None                          # global id of the current start slot

    def _write(self, frame, action=0, reward=0.0, terminated=False, truncated=False, start=None):
        slot = self.count % self.capacity
        self.frames[slot] = frame
        self.actions[slot] = action
        self.rewards[slot] = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.terminated[slot] = terminated
        self.truncated[slot] = truncated
        self.start_id[slot] = self.count if start is None else start
        self.count += 1

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        last = (self.count - 1) % self.capacity
        new_episode = (self._episode_start is None
                       or not np.array_equal(obs[-1], self.frames[last]))
        if new_episode:  # first step of a game: store its reset frame as a start slot
            self._episode_start = self.count
            self._write(obs[-1])
        self._write(next_obs[-1], action, reward, terminated, truncated, start=self._episode_start)
        if terminated or truncated:
            self._episode_start = None

    def _stacks(self, ids):
        """Rebuild the 4-frame stack ending at each global id, never crossing its start slot."""
        starts = self.start_id[ids % self.capacity]
        offsets = np.arange(-3, 1)
        stack_ids = np.maximum(ids[:, None] + offsets, starts[:, None])
        return self.frames[stack_ids % self.capacity]

    def _valid(self, ids):
        oldest = max(0, self.count - self.capacity)
        slots = ids % self.capacity
        starts = self.start_id[slots]
        not_start = starts != ids
        # Every frame used by obs (ids-4..ids-1, clipped at start) must still be in the ring.
        still_stored = np.maximum(starts, ids - 4) >= oldest
        return not_start & still_stored

    def sample(self, batch_size, rng):
        """Uniform random transitions. rng is a numpy Generator (saved with checkpoints)."""
        oldest = max(0, self.count - self.capacity)
        chosen = np.empty(0, np.int64)
        while len(chosen) < batch_size:
            ids = rng.integers(oldest + 1, self.count, size=batch_size * 2)
            chosen = np.concatenate([chosen, ids[self._valid(ids)]])
        ids = chosen[:batch_size]
        slots = ids % self.capacity
        return (self._stacks(ids - 1), self.actions[slots].astype(np.int64), self.rewards[slots],
                self._stacks(ids), self.terminated[slots], self.truncated[slots])

    def __len__(self):
        """Frames currently stored (includes one start slot per episode, so slightly above transitions)."""
        return min(self.count, self.capacity)
