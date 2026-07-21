#!/usr/bin/env python3
"""Run IIANet (liproi32x32 + sadr84 front) on preprocessed ROI/landmark npz inputs.

Expected visual inputs (already processed):
  - liproi npz: key 'data', shape [T, 32, 32]
  - lmk npz:    key 'data', shape [T, 84, 2] or [T, 84, 3] (normalized)

Default samples map to LRS3 tt mixtures:
  sample1_s1 -> 0VJqrlH9cdI_00002_0.49459_t7Xr3AsBEK4_00006_-0.49459 (s1)
  sample2_s2 -> 0ZfSOArXbGQ_00003_1.188_RplnSVTzvnU_00003_-1.188 (s2)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path("/jinzhan/lmk_avse/iianet_liproi_sadr84_front")
DEFAULT_CKPT = "ckpt/liproi32x32_sadr84_front/epoch197.pth.tar"
DEFAULT_CONFIG = "config/std_train.yml"
DEFAULT_SAMPLES_ROOT = REPO_ROOT / "samples" / "iianet_liproi_sadr84_front"

SAMPLE_SPECS = {
    "sample1_s1": {
        "utt": "0VJqrlH9cdI_00002_0.49459_t7Xr3AsBEK4_00006_-0.49459",
        "speaker": "s1",
        "gt_wav": (
            "/jinzhan/data/lrs3_rebuild/audio/wav16k/min/tt/s1/"
            "0VJqrlH9cdI_00002_0.49459_t7Xr3AsBEK4_00006_-0.49459.wav"
        ),
    },
    "sample2_s2": {
        "utt": "0ZfSOArXbGQ_00003_1.188_RplnSVTzvnU_00003_-1.188",
        "speaker": "s2",
        "gt_wav": (
            "/jinzhan/data/lrs3_rebuild/audio/wav16k/min/tt/s2/"
            "0ZfSOArXbGQ_00003_1.188_RplnSVTzvnU_00003_-1.188.wav"
        ),
    },
}


def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in model_info["model_state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)
    return model


def _align_visual(liproi, lmk, n_frames):
    while n_frames > len(liproi):
        liproi = np.insert(liproi, 0, liproi[0], axis=0)
        liproi = np.insert(liproi, -1, liproi[-1], axis=0)
        lmk = np.insert(lmk, 0, lmk[0], axis=0)
        lmk = np.insert(lmk, -1, lmk[-1], axis=0)
    return liproi[:n_frames], lmk[:n_frames]


def load_inputs(mix_wav_path, liproi_path, lmk_path):
    mix_wav, sr = sf.read(mix_wav_path, dtype="float32")
    if mix_wav.ndim > 1:
        mix_wav = mix_wav.mean(axis=-1)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz mix, got {sr} from {mix_wav_path}")

    liproi = np.load(liproi_path)["data"].astype(np.float32)
    lmk = np.load(lmk_path)["data"].astype(np.float32)
    if liproi.ndim != 3 or liproi.shape[-2:] != (32, 32):
        raise ValueError(f"liproi expects [T, 32, 32], got {liproi.shape} from {liproi_path}")
    if lmk.ndim != 3 or lmk.shape[1] != 84 or lmk.shape[2] not in (2, 3):
        raise ValueError(f"lmk expects [T, 84, 2|3], got {lmk.shape} from {lmk_path}")

    # Match inf_dataset: keep full mix length; align visual to ilen // 640.
    n_frames = mix_wav.shape[-1] // 640
    liproi, lmk = _align_visual(liproi, lmk, n_frames)

    mix_wav = torch.from_numpy(mix_wav).float()
    liproi = torch.from_numpy(liproi).float()
    lmk = torch.from_numpy(lmk[..., :2]).float()
    return mix_wav, liproi, lmk


def build_model(model_root, config_path, ckpt_path, device):
    sys.path.insert(0, str(model_root))
    import models.IIANet as module_model  # noqa: WPS433

    with open(config_path) as rfile:
        config = yaml.safe_load(rfile)

    model = module_model.IIANet(**config["iianet_kwargs"])
    model = load_pretrained_modules(model, ckpt_path)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one(model, mix_wav, liproi, lmk, device):
    mix_wav = mix_wav.to(device).unsqueeze(0)
    liproi = liproi.to(device).unsqueeze(0).unsqueeze(0)  # [B, 1, T, H, W]
    lmk = lmk.to(device).unsqueeze(0)  # [B, T, 84, 2]
    est = model(mix_wav.transpose(1, -1), liproi, lmk)  # [B, C, T]
    return est.squeeze().detach().cpu().numpy()


def resolve_sample_paths(sample_name, samples_root):
    sample_dir = Path(samples_root) / sample_name
    return {
        "name": sample_name,
        "mix": sample_dir / "mix.wav",
        "liproi": sample_dir / "liproi32x32.npz",
        "lmk": sample_dir / "lmk84.npz",
        "meta": SAMPLE_SPECS[sample_name],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="IIANet liproi32x32 + sadr84-front single-clip inference"
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Path to iianet_liproi_sadr84_front project root",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Model yaml config (default: <model-root>/config/std_train.yml)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path (default: <model-root>/ckpt/.../epoch197.pth.tar)",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=DEFAULT_SAMPLES_ROOT,
        help="Directory that contains sample1_s1 / sample2_s2 folders",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(SAMPLE_SPECS.keys()),
        choices=list(SAMPLE_SPECS.keys()),
        help="Which packaged samples to run",
    )
    parser.add_argument("--mix", type=Path, default=None, help="Custom mix wav")
    parser.add_argument("--liproi", type=Path, default=None, help="Custom liproi npz [T,32,32]")
    parser.add_argument("--lmk", type=Path, default=None, help="Custom landmark npz [T,84,2|3]")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "iianet_liproi_sadr84_front",
        help="Directory for estimated wavs",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model_root = args.model_root.resolve()
    config_path = (args.config or (model_root / DEFAULT_CONFIG)).resolve()
    ckpt_path = (args.ckpt or (model_root / DEFAULT_CKPT)).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(args.device)
    print(f"Loading model from {ckpt_path} on {device} ...", flush=True)
    model = build_model(model_root, config_path, ckpt_path, device)
    print("Model ready.", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.mix is not None or args.liproi is not None or args.lmk is not None:
        if not (args.mix and args.liproi and args.lmk):
            raise ValueError("Custom mode requires --mix, --liproi and --lmk together")
        jobs.append(
            {
                "name": args.mix.stem,
                "mix": args.mix,
                "liproi": args.liproi,
                "lmk": args.lmk,
                "meta": None,
            }
        )
    else:
        for name in args.samples:
            jobs.append(resolve_sample_paths(name, args.samples_root))

    for job in jobs:
        mix_wav, liproi, lmk = load_inputs(job["mix"], job["liproi"], job["lmk"])
        print(
            f"Inferring {job['name']}: mix={tuple(mix_wav.shape)} "
            f"liproi={tuple(liproi.shape)} lmk={tuple(lmk.shape)}",
            flush=True,
        )
        est = infer_one(model, mix_wav, liproi, lmk, device)
        out_wav = args.out_dir / f"{job['name']}.wav"
        sf.write(out_wav, est, 16000)
        msg = f"[{job['name']}] -> {out_wav}"
        if job.get("meta"):
            msg += f" (utt={job['meta']['utt']}, {job['meta']['speaker']})"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
