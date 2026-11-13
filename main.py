"""Scheduled low-rank quaternion tensor completion."""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

if sys.platform != "win32" or sys.version_info[:2] != (3, 10):
    raise RuntimeError("The bundled runtime requires 64-bit CPython 3.10 on Windows")

from utils._core import (
    initialize, update_weights, update_lowrank, pnp_input,
    update_x, update_multipliers, completed_video,
)
from utils._denoise import load_denoisers, denoise

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "akiyo": "akiyo_cif/akiyo_cif_256x256x3x20.npy",
    "carphone": "carphone_qcif/carphone_qcif_256x256x3x20.npy",
    "container": "container_cif/container_cif_256x256x3x20.npy",
    "foreman": "foreman_cif/foreman_cif_256x256x3x20.npy",
    "grandma": "grandma_qcif/grandma_qcif_256x256x3x20.npy",
    "hall": "hall_monitor_cif/hall_monitor_cif_256x256x3x20.npy",
    "news": "news_cif/news_cif_256x256x3x20.npy",
    "suzie": "suzie_qcif/suzie_qcif_256x256x3x20.npy",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="foreman")
    parser.add_argument("--input", type=Path, help="RGB video stored as a uint8 NPY array")
    parser.add_argument("--mask", type=Path, default=ROOT / "data/masks/observed_mask_mr95_seed0.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "output/recovered.npy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=200, help="maximum iteration index")
    parser.add_argument("--eta", type=float, nargs="+", default=[0.15], help="one shared weight or one weight per unfolding")
    parser.add_argument("--lambda", dest="lam", type=float, default=30.0/255**2)
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=1.08)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--p-min", type=float, default=0.30)
    parser.add_argument("--p-max", type=float, default=0.90)
    args = parser.parse_args()
    if args.K < 1 or args.mu <= 0 or args.rho <= 1 or args.lam < 0:
        parser.error("K >= 1, mu > 0, rho > 1, and lambda >= 0 are required")
    if args.gamma <= 0 or args.epsilon <= 0 or any(v <= 0 for v in args.eta):
        parser.error("gamma, epsilon, and all eta weights must be positive")
    if not 0 < args.p_min < args.p_max < 1:
        parser.error("require 0 < p-min < p-max < 1")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    path = args.input or ROOT / "data" / DATASETS[args.dataset]
    video = np.load(path, allow_pickle=False)
    if video.ndim != 4 or video.shape[1:3] != (256, 256) or video.shape[3] != 3 or video.dtype != np.uint8:
        raise ValueError("Input must be a uint8 RGB array with shape [T, 256, 256, 3]")
    mask = np.load(args.mask, allow_pickle=False)
    if mask.shape != video.shape[:3] or not np.isin(mask, (0, 1)).all():
        raise ValueError("Mask must contain 0/1 and have shape [T, 256, 256]")
    reference = torch.from_numpy(video.copy()).to(device=device, dtype=torch.float32).div_(255.0)
    omega = torch.from_numpy(mask.copy()).to(device=device, dtype=torch.bool)
    observed = reference * omega[..., None]
    state = initialize(observed, omega, args.mu)
    models = load_denoisers(ROOT, device)
    count = len(state.copies)
    eta = args.eta * count if len(args.eta) == 1 else args.eta
    if len(eta) != count:
        raise ValueError(f"Expected one eta or {count} unfolding weights")

    # Algorithm: k = 0, ..., K; X, M, V, Y1, Y2 and mu are initialized above.
    for k in range(args.K + 1):
        weights = update_weights(state, args.gamma, args.epsilon)
        update_lowrank(state, weights, eta)
        probability = args.p_min + (args.p_max-args.p_min) * (1+math.cos(math.pi*k/args.K)) / 2
        choice = 0 if random.random() <= probability else 1
        sigma = math.sqrt(args.lam / state.mu)
        state.v = denoise(pnp_input(state), sigma, choice, models)
        update_x(state)
        update_multipliers(state, args.rho)
        print(f"k={k} p={probability:.6f} denoiser={'FFDNet' if choice == 0 else 'FastDVDnet'}", flush=True)

    result = completed_video(state).detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        np.save(file, result)
    print(f"Saved final RGB video [T,H,W,3], float32: {args.output}")


if __name__ == "__main__":
    main()
