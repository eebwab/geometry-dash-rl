# Geometry Dash RL Agent

A reinforcement learning agent that plays **Geometry Dash – Stereo Madness** in real-time using raw pixel inputs and emulated keystrokes. No internal game state APIs are used.

## Architecture

The project is structured as a two-phase pipeline:

### Phase 1 — Prototype (current)
Off-the-shelf algorithms via `stable-baselines3` (DQN) wired to a custom Gymnasium environment.

| Module | Responsibility |
|---|---|
| `vision.py` | `mss` screen capture → grayscale 84×84 frames → 4-frame stack |
| `controls.py` | `pynput` keyboard actuation (cross-platform) |
| `env.py` | `gymnasium.Env` wrapper: `reset()`, `step()`, death detection |
| `config.py` | All tunable hyperparameters as frozen dataclasses |
| `train.py` | SB3 DQN training loop, TensorBoard, checkpointing |
| `calibrate.py` | Interactive utility to lock in BlueStacks window coords |

### Phase 2 — Custom Deep Q-Learning (planned)
Replace SB3 with a hand-rolled CNN + replay buffer + training loop in pure PyTorch. The Gymnasium env and vision pipeline carry over unchanged.

## Quickstart

```bash
# 1. Create virtualenv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Calibrate BlueStacks window position
python calibrate.py

# 3. Start training
python train.py
```

TensorBoard logs land in `runs/`, model checkpoints in `models/`.

## Observation Space

Frames are preprocessed to `(4, 84, 84)` uint8 — four stacked grayscale frames at 84×84 resolution, matching the classic Atari DQN setup. The stack gives the agent implicit velocity/momentum information.

## Action Space

| Action | Effect |
|---|---|
| `0` | Do nothing |
| `1` | Jump (spacebar) |

## Reward Function

| Event | Reward |
|---|---|
| Survive one frame | +1.0 |
| Death detected | −100.0 |

## Requirements

- Python 3.11+
- BlueStacks (or any Android emulator exposing a capturable window)
- Geometry Dash installed and sitting on Stereo Madness ready to play
