#!/usr/bin/env python3
"""Standalone inference for the lightweight dual-branch visual front-end.

Expected preprocessed visual inputs (npz, key 'data'):
  - roi.npz: shape [T, 32, 32]
  - lmk.npz: shape [T, 84, 2] or [T, 84, 3] (already normalized)

Default packaged samples live under samples/sample1_s1 and samples/sample2_s2.

Example:
  python inference.py
  python inference.py --mix path/mix.wav --roi path/roi.npz --lmk path/lmk.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config.yml"
DEFAULT_CKPT = REPO_ROOT / "model.pth.tar"
DEFAULT_SAMPLES_ROOT = REPO_ROOT / "samples"
DEFAULT_SAMPLES = ("sample1_s1", "sample2_s2")


def load_checkpoint(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in model_info["model_state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)
    return model


def _align_visual(roi, lmk, n_frames):
    while n_frames > len(roi):
        roi = np.insert(roi, 0, roi[0], axis=0)
        roi = np.insert(roi, -1, roi[-1], axis=0)
        lmk = np.insert(lmk, 0, lmk[0], axis=0)
        lmk = np.insert(lmk, -1, lmk[-1], axis=0)
    return roi[:n_frames], lmk[:n_frames]


def load_inputs(mix_wav_path, roi_path, lmk_path):
    mix_wav, sr = sf.read(mix_wav_path, dtype="float32")
    if mix_wav.ndim > 1:
        mix_wav = mix_wav.mean(axis=-1)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz mix, got {sr} from {mix_wav_path}")

    roi = np.load(roi_path)["data"].astype(np.float32)
    lmk = np.load(lmk_path)["data"].astype(np.float32)
    if roi.ndim != 3 or roi.shape[-2:] != (32, 32):
        raise ValueError(f"roi expects [T, 32, 32], got {roi.shape} from {roi_path}")
    if lmk.ndim != 3 or lmk.shape[1] != 84 or lmk.shape[2] not in (2, 3):
        raise ValueError(f"lmk expects [T, 84, 2|3], got {lmk.shape} from {lmk_path}")

    n_frames = mix_wav.shape[-1] // 640
    roi, lmk = _align_visual(roi, lmk, n_frames)

    mix_wav = torch.from_numpy(mix_wav).float()
    roi = torch.from_numpy(roi).float()
    lmk = torch.from_numpy(lmk[..., :2]).float()
    return mix_wav, roi, lmk


def build_model(config_path, ckpt_path, device):
    sys.path.insert(0, str(REPO_ROOT))
    from models.network import Model  # noqa: WPS433

    with open(config_path) as rfile:
        config = yaml.safe_load(rfile)

    model = Model(**config["model_kwargs"])
    model = load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one(model, mix_wav, roi, lmk, device):
    mix_wav = mix_wav.to(device).unsqueeze(0)
    roi = roi.to(device).unsqueeze(0).unsqueeze(0)  # [B, 1, T, H, W]
    lmk = lmk.to(device).unsqueeze(0)  # [B, T, 84, 2]
    est = model(mix_wav.transpose(1, -1), roi, lmk)  # [B, C, T]
    return est.squeeze().detach().cpu().numpy()


def resolve_sample_paths(sample_name, samples_root):
    sample_dir = Path(samples_root) / sample_name
    return {
        "name": sample_name,
        "mix": sample_dir / "mix.wav",
        "roi": sample_dir / "roi.npz",
        "lmk": sample_dir / "lmk.npz",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight visual front-end inference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(DEFAULT_SAMPLES),
        choices=list(DEFAULT_SAMPLES),
    )
    parser.add_argument("--mix", type=Path, default=None, help="Custom mix wav")
    parser.add_argument("--roi", type=Path, default=None, help="Custom roi npz [T,32,32]")
    parser.add_argument("--lmk", type=Path, default=None, help="Custom landmark npz [T,84,2|3]")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    ckpt_path = args.ckpt.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(args.device)
    print(f"Loading model from {ckpt_path} on {device} ...", flush=True)
    model = build_model(config_path, ckpt_path, device)
    print("Model ready.", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.mix is not None or args.roi is not None or args.lmk is not None:
        if not (args.mix and args.roi and args.lmk):
            raise ValueError("Custom mode requires --mix, --roi and --lmk together")
        jobs.append({"name": args.mix.stem, "mix": args.mix, "roi": args.roi, "lmk": args.lmk})
    else:
        for name in args.samples:
            jobs.append(resolve_sample_paths(name, args.samples_root))

    for job in jobs:
        mix_wav, roi, lmk = load_inputs(job["mix"], job["roi"], job["lmk"])
        print(
            f"Inferring {job['name']}: mix={tuple(mix_wav.shape)} "
            f"roi={tuple(roi.shape)} lmk={tuple(lmk.shape)}",
            flush=True,
        )
        est = infer_one(model, mix_wav, roi, lmk, device)
        out_wav = args.out_dir / f"{job['name']}.wav"
        sf.write(out_wav, est, 16000)
        print(f"[{job['name']}] -> {out_wav}", flush=True)


if __name__ == "__main__":
    main()
