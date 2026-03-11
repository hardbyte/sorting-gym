"""Visualize a sorting agent's strategy at different training stages as a GIF.

Trains PPO on the sorting environment, periodically capturing full episodes
rendered as text-art frames. Combines them into an animated GIF.

Requires: pip install stable-baselines3 pillow

Usage:
    python examples/visualize_training.py
    python examples/visualize_training.py --timesteps 200000 --output sorting_progress.gif
"""

import argparse
from io import BytesIO

from gymnasium.wrappers import FlattenObservation
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.wrappers import MultiDiscreteActionSpaceWrapper


def wrap_env(env):
    return FlattenObservation(MultiDiscreteActionSpaceWrapper(env))


def text_to_image(text, title="", width=520, font_size=14, padding=12):
    """Render a multi-line text string to a PIL Image."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size + 2)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    lines = text.split("\n")
    line_height = font_size + 4
    title_height = (font_size + 8) if title else 0
    height = title_height + padding * 2 + line_height * len(lines)

    img = Image.new("RGB", (width, height), color=(24, 24, 32))
    draw = ImageDraw.Draw(img)

    if title:
        draw.text((padding, padding // 2), title, fill=(100, 200, 255), font=title_font)

    y = padding + title_height
    for line in lines:
        # Colour the SORTED! marker green
        if "SORTED!" in line:
            draw.text((padding, y), line, fill=(80, 220, 80), font=font)
        elif line.startswith("Array:"):
            draw.text((padding, y), line, fill=(220, 220, 220), font=font)
        elif "v0" in line or "v1" in line or "v2" in line:
            draw.text((padding, y), line, fill=(255, 180, 80), font=font)
        else:
            # Bar chars
            draw.text((padding, y), line, fill=(80, 180, 255), font=font)
        y += line_height
    return img


def capture_episode(model, env_factory, max_steps=80):
    """Run one episode and return list of (step, rendered_text) tuples."""
    raw_env = env_factory()
    wrapped_env = wrap_env(env_factory())
    # Sync seeds so they generate the same array
    seed = 42
    raw_env.reset(seed=seed)
    obs, _ = wrapped_env.reset(seed=seed)
    # Copy the array from raw_env into wrapped to ensure same data
    inner = wrapped_env
    while hasattr(inner, "env"):
        inner = inner.env
    raw_env.A = list(inner.A)
    raw_env.v[:] = inner.v[:]

    frames = []
    frame_text = raw_env.render(mode="rgb_array")
    frames.append((0, frame_text))

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        # Sync raw env state for rendering
        raw_env.A = list(inner.A)
        raw_env.v[:] = inner.v[:]
        frame_text = raw_env.render(mode="rgb_array")
        frames.append((step, frame_text))
        if terminated or truncated:
            break

    raw_env.close()
    wrapped_env.close()
    return frames


class SnapshotCallback(BaseCallback):
    """Capture episode snapshots at specified training milestones."""

    def __init__(self, milestones, env_factory, verbose=0):
        super().__init__(verbose)
        self.milestones = sorted(milestones)
        self.env_factory = env_factory
        self.snapshots = {}  # milestone -> list of (step, text) frames
        self._next_idx = 0

    def _on_step(self) -> bool:
        if self._next_idx >= len(self.milestones):
            return True
        target = self.milestones[self._next_idx]
        if self.num_timesteps >= target:
            print(f"  Capturing snapshot at {self.num_timesteps} timesteps...")
            frames = capture_episode(self.model, self.env_factory)
            self.snapshots[target] = frames
            self._next_idx += 1
        return True


def build_gif(all_stage_frames, output_path, fps=4):
    """Build an animated GIF from staged episode frames.

    all_stage_frames: list of (title, [(step, text), ...])
    """
    images = []

    for title, frames in all_stage_frames:
        # Add a title card
        title_img = text_to_image("", title=title, width=520)
        for _ in range(fps):  # hold title for 1 second
            images.append(title_img)

        for step, text in frames:
            img = text_to_image(text, title=f"{title}  |  Step {step}")
            images.append(img)

        # Hold last frame longer
        if images:
            for _ in range(fps):
                images.append(images[-1])

    if not images:
        print("No frames captured!")
        return

    # Normalize all images to same size
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

    duration = int(1000 / fps)
    normalized[0].save(
        output_path,
        save_all=True,
        append_images=normalized[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved GIF to {output_path} ({len(normalized)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize sorting agent training progress")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--output", type=str, default="docs/sorting_progress.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second in GIF")
    args = parser.parse_args()

    env_factory = lambda: BasicNeuralSortInterfaceEnv(k=3)

    # Milestones to capture snapshots
    milestones = [0, args.timesteps // 10, args.timesteps // 4, args.timesteps // 2, args.timesteps]

    print(f"Training PPO on sorting env for {args.timesteps} timesteps")
    print(f"Will capture snapshots at: {milestones}")

    env = wrap_env(env_factory())

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
    )

    # Capture untrained agent first
    print("  Capturing snapshot at 0 timesteps (untrained)...")
    untrained_frames = capture_episode(model, env_factory)

    callback = SnapshotCallback(milestones=[m for m in milestones if m > 0], env_factory=env_factory)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    env.close()

    # Build combined frame list
    all_stages = [("Untrained (0 steps)", untrained_frames)]
    for m in milestones:
        if m > 0 and m in callback.snapshots:
            label = f"{m // 1000}k steps"
            all_stages.append((label, callback.snapshots[m]))

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    build_gif(all_stages, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
