"""Visualize sorting and knapsack agents at different training stages as GIFs.

Trains PPO, captures episodes at milestones, renders text-art frames to images,
and assembles animated GIFs showing the agent's strategy evolving.

Requires: pip install stable-baselines3 pillow

Usage:
    python examples/visualize_training.py
    python examples/visualize_training.py --env sort --timesteps 200000
    python examples/visualize_training.py --env knapsack --timesteps 200000
"""

import argparse
import os

import torch
# Avoid hangs in constrained environments (CI, containers)
torch.set_num_threads(1)

from gymnasium.wrappers import FlattenObservation
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.knapsack import KnapsackEnv
from sorting_gym.envs.wrappers import MultiDiscreteActionSpaceWrapper


def wrap_env(env):
    return FlattenObservation(MultiDiscreteActionSpaceWrapper(env))


def get_inner_env(wrapped):
    """Walk the wrapper chain to get the actual environment."""
    env = wrapped
    while hasattr(env, "env"):
        env = env.env
    return env


# ---------------------------------------------------------------------------
# Text -> Image rendering
# ---------------------------------------------------------------------------

def text_to_image(text, title="", width=560, font_size=14, padding=12):
    """Render multi-line text to a PIL Image with a dark background."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size + 2
        )
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    lines = text.split("\n")
    line_height = font_size + 4
    title_height = (font_size + 10) if title else 0
    height = title_height + padding * 2 + line_height * len(lines)

    img = Image.new("RGB", (width, height), color=(24, 24, 32))
    draw = ImageDraw.Draw(img)

    if title:
        draw.text((padding, padding // 2), title, fill=(100, 200, 255), font=title_font)

    y = padding + title_height
    for line in lines:
        if "SORTED!" in line or "DONE!" in line:
            color = (80, 220, 80)
        elif line.startswith("Array:") or line.startswith("Capacity:") or line.startswith("Value"):
            color = (220, 220, 220)
        elif "v0" in line or "v1" in line or "v2" in line or "v3" in line:
            color = (255, 180, 80)
        elif "\u2713" in line:
            color = (80, 220, 80)
        elif "\u2588" in line:
            color = (80, 180, 255)
        else:
            color = (180, 180, 190)
        draw.text((padding, y), line, fill=color, font=font)
        y += line_height
    return img


# ---------------------------------------------------------------------------
# Episode capture — works directly on the wrapped env
# ---------------------------------------------------------------------------

def capture_episode(model, env_factory, seed=42, max_steps=60, max_frames=12):
    """Run one episode and return a list of (step, rendered_text) tuples.

    Only keeps frames where the visual state changed (swaps, selections,
    pointer moves). Subsamples to max_frames, always keeping first and last.
    """
    wrapped = wrap_env(env_factory())
    inner = get_inner_env(wrapped)

    obs, _ = wrapped.reset(seed=seed)
    all_frames = [(0, inner.render(mode="rgb_array"))]
    prev_text = all_frames[0][1]
    seen_texts = {prev_text}

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        text = inner.render(mode="rgb_array")
        # Keep frames with new visual states (skip repeating patterns)
        if text != prev_text and text not in seen_texts:
            all_frames.append((step, text))
            seen_texts.add(text)
            prev_text = text
        elif text != prev_text:
            prev_text = text
        if terminated or truncated:
            # Always capture the terminal frame
            if all_frames[-1][1] != text:
                all_frames.append((step, text))
            break

    wrapped.close()

    # Subsample to max_frames, always keeping first and last
    if len(all_frames) > max_frames:
        indices = {0, len(all_frames) - 1}
        step_size = (len(all_frames) - 1) / (max_frames - 1)
        for i in range(max_frames):
            indices.add(min(int(i * step_size), len(all_frames) - 1))
        all_frames = [all_frames[i] for i in sorted(indices)]

    return all_frames


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------

def build_gif(all_stage_frames, output_path, frame_duration_ms=300):
    """Build an animated GIF from staged episode frames.

    all_stage_frames: list of (title, [(step, text), ...])
    """
    images = []
    durations = []

    for title, frames in all_stage_frames:
        # Title card
        title_img = text_to_image("\n\n", title=title)
        images.append(title_img)
        durations.append(1200)  # hold title 1.2s

        for i, (step, text) in enumerate(frames):
            img = text_to_image(text, title=f"{title}  \u2502  Step {step}")
            images.append(img)
            # Hold last frame of each stage longer
            if i == len(frames) - 1:
                durations.append(1500)
            else:
                durations.append(frame_duration_ms)

    if not images:
        print("No frames captured!")
        return

    # Normalize all images to the same size
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    normalized = []
    for img in images:
        if img.size != (max_w, max_h):
            new_img = Image.new("RGB", (max_w, max_h), color=(24, 24, 32))
            new_img.paste(img, (0, 0))
            normalized.append(new_img)
        else:
            normalized.append(img)

    normalized[0].save(
        output_path,
        save_all=True,
        append_images=normalized[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved {output_path} ({len(normalized)} frames, {os.path.getsize(output_path) // 1024}KB)")


# ---------------------------------------------------------------------------
# Training + snapshot callback
# ---------------------------------------------------------------------------

class SnapshotCallback(BaseCallback):
    """Capture episode snapshots at training milestones."""

    def __init__(self, milestones, env_factory, verbose=0):
        super().__init__(verbose)
        self.milestones = sorted(milestones)
        self.env_factory = env_factory
        self.snapshots = {}
        self._next_idx = 0

    def _on_step(self) -> bool:
        if self._next_idx >= len(self.milestones):
            return True
        target = self.milestones[self._next_idx]
        if self.num_timesteps >= target:
            print(f"  Snapshot at {self.num_timesteps:,} timesteps...")
            # Use a consistent seed per milestone for reproducibility
            seed = 42 + self._next_idx
            self.snapshots[target] = capture_episode(
                self.model, self.env_factory, seed=seed
            )
            self._next_idx += 1
        return True


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "sort": {
        "factory": lambda: BasicNeuralSortInterfaceEnv(k=3),
        "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        "ppo_kwargs": dict(
            learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=8,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.05, clip_range=0.2,
        ),
        "default_timesteps": 200_000,
        "output": "docs/sorting_progress.gif",
    },
    "knapsack": {
        "factory": lambda: KnapsackEnv(k=4, base=20, starting_min_items=4, capacity_ratio=0.5),
        "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        "ppo_kwargs": dict(
            learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.05, clip_range=0.2,
        ),
        "default_timesteps": 100_000,
        "output": "docs/knapsack_progress.gif",
    },
}


def train_and_visualize(env_name, timesteps=None, output_override=None):
    config = ENV_CONFIGS[env_name]
    factory = config["factory"]
    output = output_override or config["output"]
    timesteps = timesteps or config.get("default_timesteps", 100_000)

    milestones = [
        timesteps // 10,
        timesteps // 4,
        timesteps // 2,
        timesteps,
    ]

    print(f"\nTraining PPO on {env_name} for {timesteps:,} timesteps")
    print(f"Snapshots at: {[f'{m:,}' for m in milestones]}")

    env = wrap_env(factory())
    model = PPO(
        "MlpPolicy", env, verbose=0,
        policy_kwargs=config["policy_kwargs"],
        **config["ppo_kwargs"],
    )

    # Untrained snapshot
    print("  Snapshot at 0 timesteps (untrained)...")
    untrained = capture_episode(model, factory)

    cb = SnapshotCallback(milestones, factory)
    model.learn(total_timesteps=timesteps, callback=cb)
    env.close()

    # Assemble stages
    stages = [("Untrained (0 steps)", untrained)]
    for m in milestones:
        if m in cb.snapshots:
            if m >= 1000:
                label = f"{m // 1000}k steps"
            else:
                label = f"{m} steps"
            stages.append((label, cb.snapshots[m]))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    build_gif(stages, output, frame_duration_ms=350)


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress as animated GIFs")
    parser.add_argument("--env", choices=list(ENV_CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Training timesteps (default: per-env, 200k sort, 100k knapsack)")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    envs = list(ENV_CONFIGS.keys()) if args.env == "all" else [args.env]
    for env_name in envs:
        train_and_visualize(env_name, args.timesteps, args.output)


if __name__ == "__main__":
    main()
