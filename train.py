"""Phase 1 training entry point using Stable Baselines3 DQN."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from env import GeometryDashEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def make_env(config: Config, log_dir: Path) -> DummyVecEnv:
    """Wrap the custom env for SB3 (Monitor + VecEnv).

    Note: Frame stacking is handled inside GeometryDashEnv (4x84x84 obs).
    Do NOT add VecFrameStack here — that would double-stack frames.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    def _init() -> Monitor:
        env = GeometryDashEnv(config)
        return Monitor(env, filename=str(log_dir / "monitor.csv"))

    return DummyVecEnv([_init])


def train(config: Config | None = None, resume_path: str | None = None) -> None:
    config = config or Config()
    train_cfg = config.train

    log_dir = Path(train_cfg.log_dir)
    model_dir = Path(train_cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config, log_dir / "env_logs")

    if resume_path and Path(resume_path).exists():
        logger.info("Resuming model from %s", resume_path)
        model = DQN.load(resume_path, env=env, tensorboard_log=str(log_dir))
    else:
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=train_cfg.learning_rate,
            buffer_size=train_cfg.buffer_size,
            learning_starts=train_cfg.learning_starts,
            batch_size=train_cfg.batch_size,
            gamma=train_cfg.gamma,
            train_freq=train_cfg.train_freq,
            target_update_interval=train_cfg.target_update_interval,
            exploration_fraction=train_cfg.exploration_fraction,
            exploration_final_eps=train_cfg.exploration_final_eps,
            tensorboard_log=str(log_dir),
            seed=train_cfg.seed,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=str(model_dir),
        name_prefix="dqn_gd",
        save_replay_buffer=True,
    )

    logger.info("Starting training for %d timesteps", train_cfg.total_timesteps)
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = model_dir / "dqn_gd_final"
    model.save(str(final_path))
    logger.info("Model Checkpoint Saved: %s", final_path)

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Geometry Dash DQN agent (Phase 1)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip to resume")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--left", type=int, default=None, help="Capture region left (px)")
    parser.add_argument("--top", type=int, default=None, help="Capture region top (px)")
    parser.add_argument("--width", type=int, default=None, help="Capture region width (px)")
    parser.add_argument("--height", type=int, default=None, help="Capture region height (px)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    if args.timesteps is not None:
        config.train.total_timesteps = args.timesteps
    if args.log_dir is not None:
        config.train.log_dir = args.log_dir

    cap = config.capture
    if args.left is not None:
        cap.left = args.left
    if args.top is not None:
        cap.top = args.top
    if args.width is not None:
        cap.width = args.width
    if args.height is not None:
        cap.height = args.height

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
