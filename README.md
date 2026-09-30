# SWIFT-MAN (Taylor's Version)

### A Deep Q-Network learns Ms. Pac-Man, and the footage is repainted as a psychedelic 8-bit Taylor Swift fever dream

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebright09/swift-man-dqn-class3/blob/main/pacman_dqn.ipynb)

![SWIFT-MAN gameplay](results/final_best.gif)

**The real part:** a reinforcement-learning agent (DQN) that learns to play Atari Ms. Pac-Man from raw pixels.
**The unhinged part:** every recorded clip replaces Ms. Pac-Man with a Mario/Luigi-style 8-bit Taylor Swift. The pellets become tiny Swifties and bags of money, and the ghosts become HATERS, PAPARAZZI, TICKET BOTS, and CRITICS. All of it plays on a rainbow plasma maze with CRT glitches, screen shake, and a fake net-worth counter.

> **Parody disclaimer:** this is a fan-made classroom joke. It is not affiliated with, endorsed by, or known to Taylor Swift, her team, her label, or Atari. No Swifties were harmed. They were *recruited*.

Fundamentals of AI, Fall 2026, Class 3. Built on the course's original notebook ([pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn)).

---

## The one fact that matters: the glitter is cosmetic

| | The neural network sees | You see |
|---|---|---|
| Hero | a gray smudge | 8-bit Taylor: bangs, cat-eye liner, red lip, mic, sequins, pink boots, walk cycle, rainbow trail |
| Pellets | small gray smudges | tiny Swifties with friendship bracelets, and money bags |
| Power pellets | slightly bigger smudges | giant money bags with spinning rainbow rays |
| Ghosts | darker smudges | HATERS / PAPARAZZI / TICKET BOTS / CRITICS, in sunglasses, firing camera flashes |
| Maze | 84 × 84 grayscale | rainbow plasma with chromatic aberration, torn scanlines, and random color inversion |

The makeover (`SwiftSkin` in section 3½ of the notebook) only runs on frames saved to GIFs, after each move has already been chosen from the plain game screen. **Training is unchanged.** A 5-episode run with the same seed produced exactly the same decisions, learning updates, training scores, and before/after scores as the original notebook.

Set `SWIFT_MODE = False` in the notebook to get the original beige Atari footage back.

## How to run it

Open **[pacman_dqn.ipynb](pacman_dqn.ipynb)**, choose three numbers in section 1, and Run All.

| Choice | What it controls | Course starting point | This run |
|---|---|---|---|
| Exploration | Fraction of random training moves after warm-up | `0.20` | `0.10` |
| Episodes | Number of training games | `100` | `100` |
| Learning rate | Size of each learning update | `0.0001` | `0.00025` |

**Colab:** click the badge above, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all.

**Local (Mac/Linux):**

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab pacman_dqn.ipynb
```

Try 5 episodes first as a quick check. On an M1 MacBook Pro, 5 episodes take under a minute and 100 episodes take about 9 minutes (Apple MPS).

## Results

100 episodes, 10% exploration, learning rate 0.00025, on an M1 MacBook Pro (MPS): 60,743 decisions, 14,936 learning updates, 8.6 minutes.

| Evaluation game (seed) | Before training | After 100 games |
|---|---|---|
| 101 | 350 | 1,210 |
| 202 | 500 | 700 |
| 303 | 320 | 720 |
| 404 | 800 | 1,280 |
| 505 | 490 | 430 |
| **Mean** | **492** | **868 (+376)** |

![Training dashboard](results/training_dashboard.png)

| Before training (debut era) | After 100 games |
|---|---|
| ![before](results/before_training.gif) | ![after](results/after_100_games.gif) |

What the numbers do and don't say:
- Four of the five evaluation games improved and one got slightly worse. Five games is a small sample.
- The training-game average barely moved (roughly 680 → 730 per 25-game block). That's partly because 10% of training moves are random, and partly because 100 games is tiny by Atari standards: the original DQN paper trained for about 50 million frames, and this run used about 240,000.
- The loss *rose* over training. That's normal for DQN: as the network starts predicting bigger future rewards, its targets grow and move, so the error isn't a scoreboard.
- Checkpoints (`.pt`) aren't committed (about 7 MB each); rerun the notebook to regenerate them.

## How the DQN works (no glitter)

- **Environment:** `ALE/MsPacman-v5`. Four emulator frames per decision, sticky actions (25%), up to 30 no-op starts, and a 3,000-decision cap per game.
- **Observation:** four stacked 84 × 84 grayscale frames, so the network can tell which way things are moving.
- **Network:** three convolution layers plus two fully connected layers, outputting one **Q-value** (expected future points) per joystick move.
- **Exploration:** epsilon-greedy. 1,000 random warm-up decisions, then a constant epsilon (10% here).
- **Learning:** experience replay (5,000 transitions), batches of 32 every 4 decisions, a target network synced every 1,000 decisions, gamma 0.99, Huber loss, and Adam. Training rewards are clipped to [-1, 1]; every reported score is the raw game score.
- **Evaluation:** the same five seeds before and after training, with 5% exploration, on a separate environment that never touches replay memory or weights.

## How the makeover works

- **Finding things:** OpenCV connected components on the full-color Atari frame. Small wall-colored blobs are pellets, taller ones are power pellets, the yellow blob is Ms. Pac-Man, and 8 × 10 blobs in ghost colors are ghosts.
- **Sprites:** hand-written text maps, one letter per pixel, drawn as 2 × 2 blocks. That's how NES sprites were made. Taylor is a 14 × 18 side-view sprite with two walk/chomp frames. She flips to face the way she's moving and gets a glowing halo so you can find her.
- **Chaos:** an HSV plasma recomputed every frame, a neon maze with outlines, RGB channel offsets, glitch slices every ninth frame, color inversion every 29th, shake on ghost eats, a strobe on power pellets, and CRT scanlines.
- **HUD:** score, a "net worth" of $1.6B plus $1M per point (a bit), Swifties recruited, bags secured, a rotating RL-themed caption, and a scrolling ticker.
- **Cost:** about 40 ms per frame, only on the ~75 frames per clip. Each GIF is about 4.6 MB because rainbow plasma compresses terribly.

## What's in this repo

| Path | What it is |
|---|---|
| `pacman_dqn.ipynb` | The whole project, self-contained (Colab-ready), with no saved outputs |
| `results/` | Evidence from the 100-episode run: GIFs, dashboard, scores, config, training log |
| `pacman_player.py` | Optional local popup GIF player |
| `tests/verify_notebook.py` | Runs every cell in order with 5 episodes and checks every artifact |
| `requirements.txt` | Local dependencies |

Run the test with `python tests/verify_notebook.py --kernel <your-kernel> --no-popups`.

## Built with AI

This project was built with **Claude Code** (Anthropic) at my direction, starting from the instructor's DQN notebook. Claude Code wrote the SWIFT-MAN makeover, rewrote the notebook text, ran the training, and drafted this README. I chose the three hyperparameters. The reflection in the notebook is mine to write.

Review log:
1. Built the pixel-art sprites and reviewed a rendered sprite sheet. Switched to a Mario/Luigi-style 8-bit Taylor at my request.
2. Rendered real game frames and fixed ghosts being painted over by the plasma.
3. Ran `tests/verify_notebook.py` end to end: PASS. Confirmed the 5-episode training and evaluation numbers match the original notebook exactly.
4. Fixed a ticker glyph the default font can't draw, and added a halo so Taylor stays visible.
5. Ran the full 100-episode experiment and copied the evidence into `results/`.

## Sources

- [ALE installation](https://ale.farama.org/getting-started/)
- [Gymnasium Atari preprocessing](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing)
- [Gymnasium frame stacking](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStackObservation)
- [DQN paper: Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- Original classroom notebook: [pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn)
