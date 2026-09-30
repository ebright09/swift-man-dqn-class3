"""Turn Ms. Pac-Man into the observations the network learns from.

Exactly one frame-skip mechanism: the base ALE environment is created with frameskip=1
and AtariPreprocessing does the skipping (4 frames per decision, max-pooled over the last
two to remove Atari flicker). Combining the v5 default frameskip=4 with the wrapper's
frame_skip=4 would skip 16 frames per decision; Gymnasium 1.3 refuses it with a ValueError.
"""
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  # makes the "ALE/..." ids available; no AutoROM needed

ENV_ID = "ALE/MsPacman-v5"
WRAPPER_SETTINGS = {
    "env_id": ENV_ID,
    "base_frameskip": 1,               # base env does not skip...
    "frame_skip": 4,                   # ...AtariPreprocessing skips 4 frames per step
    "sticky_action_probability": 0.25, # ALE v5 default: 25% chance the last action repeats
    "noop_max": 30,                    # up to 30 do-nothing steps at reset, for varied starts
    "screen_size": 84,                 # resize to 84x84
    "grayscale": True,
    "scale_obs": False,                # keep uint8 0-255; the model scales on the device
    "terminal_on_life_loss": False,    # losing one life does not end the episode
    "frame_stack": 4,
    "stack_padding": "reset",          # after reset, the stack is 4 copies of the first frame
}


def make_env(max_steps):
    """Build the wrapped environment. render() still returns full-color RGB frames for GIFs."""
    s = WRAPPER_SETTINGS
    env = gym.make(ENV_ID, frameskip=s["base_frameskip"],
                   repeat_action_probability=s["sticky_action_probability"], render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=s["noop_max"], frame_skip=s["frame_skip"], screen_size=s["screen_size"],
        terminal_on_life_loss=s["terminal_on_life_loss"], grayscale_obs=s["grayscale"],
        scale_obs=s["scale_obs"])
    # The stack is rebuilt on every reset, so frames from two games are never stacked together.
    env = gym.wrappers.FrameStackObservation(env, stack_size=s["frame_stack"], padding_type=s["stack_padding"])
    # TimeLimit reports truncated=True (not terminated) at the cap, so learning can tell them apart.
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
