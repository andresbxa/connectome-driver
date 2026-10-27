"""Record a video of a trained policy driving, next to what it sees and its readout activity.

Three panels per frame: the track (full-resolution render), the network's input (the last 4
grayscale 84x84 frames) with the action probabilities, and a scrolling raster of the readout
units (for a graph core, the descending neurons) over the last 100 agent steps.

The episode is a typical one, not the best: by default the evaluation track (seed 10_000 + i,
as in evaluate.py) whose return in ``eval.json`` is closest to the median. The action sampling
is seeded, and the return shown is the one of this replay. Writes an MP4 at the game's 50 fps,
a short GIF for the README and a JSON with what was recorded. The track is rendered at full
resolution on every simulated frame for the video; the policy still sees only the frames it
sees in training.

Usage:
    python scripts/record_demo.py runs/connectome/seed1 [--episode 7] [--gif-seconds 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from connectome_driver.data.graphs import build_graph
from connectome_driver.envs.car_racing import make_env, observation_shape
from connectome_driver.models.policy import build_policy, needs_graph

EVAL_SEED_OFFSET = 10_000  # same tracks as scripts/evaluate.py
ACTIONS = ["nothing", "left", "right", "gas", "brake"]  # CarRacing discrete actions
FPS = 50  # CarRacing simulation rate
WINDOW = 100  # agent steps shown in the raster

# Layout (pixels). The track render is 600x400.
HEADER, FOOTER, TRACK_W, TRACK_H, SIDE_W = 44, 30, 600, 400, 420
WIDTH, HEIGHT = TRACK_W + SIDE_W, HEADER + TRACK_H + FOOTER
OBS_TILE, RASTER_W, RASTER_H = 90, 400, 150

BG = (20, 22, 28)
INK = (230, 232, 238)
MUTED = (150, 154, 166)
ACCENT = (42, 120, 214)  # connectome color in plot_curves.py
BAR_BG = (48, 52, 62)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ramp_lut() -> np.ndarray:
    """Single-hue sequential ramp, dark (inactive) to light (active), as a 256x3 table."""
    stops = np.array([[0.0, *BG], [0.55, *ACCENT], [1.0, 214, 232, 255]])
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, stops[:, 0], stops[:, c]) for c in (1, 2, 3)], 1).astype(np.uint8)


def text(img, s, x, y, scale=0.45, color=INK, thickness=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def pick_episode(run_dir: Path) -> tuple[int, float | None]:
    path = run_dir / "eval.json"
    if not path.exists():
        return 0, None
    returns = np.array(json.loads(path.read_text())["returns"])
    median = float(np.median(returns))
    return int(np.argmin(np.abs(returns - median))), median


def core_labels(cfg, graph) -> tuple[str, str]:
    core = cfg.model.core.type
    if core == "graph":
        variant = cfg.model.graph.get("variant", "connectome")  # runs older than Phase 4 have none
        g = graph.graph
        if variant == "connectome":
            title = "PPO policy with a FlyWire-wired core"
        else:
            title = f"PPO policy with a graph core ({variant} null model)"
        foot = (f"Encoder and decoder are learned; the core's wiring is fixed "
                f"({g.num_nodes:,} neurons, {g.num_edges:,} synapse-weighted edges, T={cfg.model.core.steps})")
        return title, foot
    if core == "mlp":
        return "PPO policy with an MLP core (control)", "Same encoder, decoder and training as the connectome policy"
    return "PPO policy without a core (ablation)", "Same encoder, decoder and training as the connectome policy"


def play(env, policy, device, episode: int):
    """One episode. Returns the rendered frames, per-frame rewards and per-agent-step records."""
    base = env.unwrapped
    frames, rewards, activity = [], [], []
    original_step = base.step

    def step_and_render(action):  # every simulated frame, also those hidden by frame skip
        out = original_step(action)
        frames.append(base.render())
        rewards.append(float(out[1]))
        return out

    base.step = step_and_render
    hook = policy.core.register_forward_hook(
        lambda _m, _i, out: activity.append(out[0].float().norm(dim=-1).cpu().numpy() / out.shape[-1] ** 0.5)
    )
    torch.manual_seed(episode)
    obs, _ = env.reset(seed=EVAL_SEED_OFFSET + episode)
    frames.append(base.render())
    rewards.append(0.0)
    steps = []
    while True:
        with torch.no_grad():
            logits, _ = policy(torch.as_tensor(obs, device=device)[None])
        probs = torch.softmax(logits, -1)[0]
        action = int(torch.multinomial(probs, 1))
        steps.append({"obs": obs, "probs": probs.cpu().numpy(), "action": action})
        obs, _, terminated, truncated, info = env.step(action)
        steps[-1]["last_frame"] = len(frames) - 1
        if terminated or truncated:
            hook.remove()
            return frames, np.array(rewards), steps, np.array(activity), float(info["episode"]["r"])


def raster_image(activity: np.ndarray) -> np.ndarray:
    """(steps, units) -> (RASTER_H, steps) uint8, each unit scaled to its own 2-98 percentile
    range over the episode, units ordered by their loading on the first principal component."""
    lo, hi = np.percentile(activity, [2, 98], axis=0)
    scaled = (activity - lo) / np.maximum(hi - lo, 1e-6)
    centered = scaled - scaled.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    order = np.argsort(vt[0])
    rows = scaled[:, order].T.astype(np.float32)  # (units, steps)
    rows = cv2.resize(rows, (rows.shape[1], RASTER_H), interpolation=cv2.INTER_AREA)
    return (np.clip(rows, 0, 1) * 255).astype(np.uint8)


def compose(frame, k, steps, raster, lut, score, title, foot, track_seed, n_units, unit_label):
    img = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    img[:] = BG
    text(img, title, 12, 28, 0.62, INK, 1)
    text(img, f"track {track_seed}   score {score:6.1f}", WIDTH - 250, 28, 0.55, INK)
    img[HEADER:HEADER + TRACK_H, :TRACK_W] = frame
    text(img, foot, 12, HEIGHT - 11, 0.42, MUTED)

    x0, y0 = TRACK_W + 12, HEADER + 6
    text(img, "input: last 4 frames, 84x84 gray", x0, y0 + 12, 0.42, MUTED)
    obs = steps[k]["obs"]
    for i in range(4):
        tile = cv2.resize(obs[i], (OBS_TILE, OBS_TILE), interpolation=cv2.INTER_NEAREST)
        x, y = x0 + (i % 2) * (OBS_TILE + 4), y0 + 20 + (i // 2) * (OBS_TILE + 4)
        img[y:y + OBS_TILE, x:x + OBS_TILE] = tile[..., None]

    bx, by = x0 + 2 * OBS_TILE + 20, y0 + 12
    text(img, "action probability", bx, by, 0.42, MUTED)
    for a, (name, p) in enumerate(zip(ACTIONS, steps[k]["probs"], strict=True)):
        y = by + 16 + a * 34
        chosen = a == steps[k]["action"]
        text(img, name, bx, y + 10, 0.42, INK if chosen else MUTED)
        img[y + 16:y + 26, bx:bx + 160] = BAR_BG
        img[y + 16:y + 26, bx:bx + max(int(160 * p), 1)] = ACCENT if chosen else MUTED

    ry = HEADER + TRACK_H - RASTER_H - 8
    text(img, f"{n_units:,} {unit_label}, last {WINDOW} steps (bright = more active)", x0, ry - 8, 0.42, MUTED)
    window = np.zeros((RASTER_H, WINDOW), np.uint8)
    start = max(0, k + 1 - WINDOW)
    window[:, WINDOW - (k + 1 - start):] = raster[:, start:k + 1]
    window = np.repeat(window, RASTER_W // WINDOW, axis=1)
    img[ry:ry + RASTER_H, x0:x0 + RASTER_W] = lut[window]
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episode", type=int, default=None, help="evaluation track index (default: median)")
    parser.add_argument("--out", type=Path, default=None, help="output path without extension")
    parser.add_argument("--gif-seconds", type=float, default=12.0)
    parser.add_argument("--gif-width", type=int, default=680)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    device = torch.device(args.device)
    if device.type == "cpu" and cfg.model.core.type == "graph":
        cfg.model.core.dtype = "float32"  # bfloat16 sparse products are CUDA-only
    episode, eval_median = (args.episode, None) if args.episode is not None else pick_episode(args.run_dir)

    env = make_env(cfg.env, render_mode="rgb_array")
    graph = build_graph(cfg.model.graph, seed=cfg.seed) if needs_graph(cfg.model) else None
    policy = build_policy(cfg.model, observation_shape(cfg.env), env.action_space.n, graph).to(device)
    policy.load_state_dict(torch.load(args.run_dir / "policy.pt", map_location=device))
    policy.eval()

    frames, rewards, steps, activity, episode_return = play(env, policy, device, episode)
    env.close()
    print(f"track {EVAL_SEED_OFFSET + episode}: return {episode_return:.1f} "
          f"({len(steps)} agent steps, {len(frames)} frames)", flush=True)

    title, foot = core_labels(cfg, graph)
    unit_label = "descending neurons (readout)" if cfg.model.core.type == "graph" else "readout units"
    raster, lut, score = raster_image(activity), ramp_lut(), np.cumsum(rewards)
    step_of_frame = np.searchsorted([s["last_frame"] for s in steps], np.arange(len(frames)))
    step_of_frame = np.minimum(step_of_frame, len(steps) - 1)

    out = args.out or Path("results") / f"demo_{cfg.exp_name}_seed{cfg.seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    # MPEG-4 Part 2: the pip OpenCV wheels ship no H.264 encoder. The GIF is the README version.
    writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    gif_frames, gif_every = [], 4  # one GIF frame per agent step (12.5 fps)
    gif_height = round(HEIGHT * args.gif_width / WIDTH)
    for f, frame in enumerate(frames):
        k = int(step_of_frame[f])
        img = compose(frame, k, steps, raster, lut, score[f], title, foot,
                      EVAL_SEED_OFFSET + episode, activity.shape[1], unit_label)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if f % gif_every == 0 and f < args.gif_seconds * FPS:
            small = cv2.resize(img, (args.gif_width, gif_height), interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(small).quantize(colors=128))
    writer.release()
    gif_frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif_frames[1:],
                       duration=1000 * gif_every // FPS, loop=0, optimize=True)

    info = {
        "run": args.run_dir.as_posix(),
        "track_seed": EVAL_SEED_OFFSET + episode,
        "return": episode_return,
        "eval_median": eval_median,
        "policy": "sampled",
        "frames": len(frames),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    sizes = {s: out.with_suffix(s).stat().st_size / 1e6 for s in (".mp4", ".gif")}
    print(f"wrote {out}.mp4 ({sizes['.mp4']:.1f} MB), .gif ({sizes['.gif']:.1f} MB), .json")


if __name__ == "__main__":
    main()
