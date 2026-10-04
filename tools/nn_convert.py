#!/usr/bin/env python3
"""nn_convert -- a PyTorch checkpoint to tglab's .tgw weight format.

Development-time only. tglab never runs Python: this turns a published
checkpoint into the flat file src/algo_util/nn.h reads, once, and the result
ships beside the executable.

Every BatchNorm is FOLDED into the convolution before it. At inference a
BatchNorm is a fixed per-channel scale and shift, y = (x - mean) / sqrt(var +
eps) * gamma + beta, and a convolution followed by one is a convolution with
rescaled weights and a new bias. Folding here means the engine needs no
BatchNorm operation at all.

Folding works from the state dict alone, so no model code is needed: a
BatchNorm at "<prefix>.<n>" (it has a running_mean) follows the convolution at
"<prefix>.<n-1>" (a 4-D weight), which is how both torchvision's VGG lists
and nn.Sequential blocks are laid out.

usage:
    python tools/nn_convert.py dad models/dad.tgw [--fp32]
    python tools/nn_convert.py loma_b128 models/dedode_b128.tgw --prefix _descriptor.
    python tools/nn_convert.py loma_b128 models/loma_b128.tgw --exclude _detector. --exclude _descriptor.
    python tools/nn_convert.py <checkpoint.pth> out.tgw [--fp32]
"""
import argparse
import struct
import sys

import numpy as np
import torch

# Published checkpoints, by the name tglab's algorithms use.
MODELS = {
    "dad": "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth",
    "dedode_b": "https://github.com/Parskatt/DeDoDe/releases/download/"
                "dedode_pretrained_models/dedode_descriptor_B.pth",
    # LoMa-B128 entire: DaD (identical to "dad"), its own 128-dimensional
    # DeDoDe-B descriptor, and the matcher. Take one with --prefix.
    "loma_b128": "https://github.com/davnords/storage/releases/download/loma/loma_B128.pth",
}

BN_EPS = 1e-5   # torch.nn.BatchNorm2d's default, which every model here uses


def load_state(source, prefix="", exclude=()):
    if source in MODELS:
        sd = torch.hub.load_state_dict_from_url(MODELS[source], map_location="cpu")
    else:
        sd = torch.load(source, map_location="cpu")
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {k[len(prefix):]: v.float() for k, v in sd.items()
            if k.startswith(prefix) and not any(k.startswith(e) for e in exclude)
            and torch.is_tensor(v) and v.is_floating_point()}


def fold(sd):
    """Returns (tensors, folded): the state dict with every BatchNorm merged
    into the convolution before it, and how many were merged."""
    out = dict(sd)
    folded = 0
    for key in sorted(k for k in sd if k.endswith(".running_mean")):
        bn = key[: -len(".running_mean")]
        prefix, _, idx = bn.rpartition(".")
        if not idx.isdigit():
            continue
        conv = f"{prefix}.{int(idx) - 1}"
        w = sd.get(conv + ".weight")
        if w is None or w.dim() != 4:
            sys.exit(f"BatchNorm {bn} does not follow a convolution")
        scale = sd[bn + ".weight"] / torch.sqrt(sd[bn + ".running_var"] + BN_EPS)
        b = sd.get(conv + ".bias", torch.zeros(w.shape[0]))
        out[conv + ".weight"] = w * scale[:, None, None, None]
        out[conv + ".bias"] = (b - sd[bn + ".running_mean"]) * scale + sd[bn + ".bias"]
        for s in (".weight", ".bias", ".running_mean", ".running_var"):
            out.pop(bn + s, None)
        folded += 1
    return out, folded


def write_tgw(path, tensors, fp16):
    with open(path, "wb") as f:
        f.write(b"TGW1")
        f.write(struct.pack("<I", len(tensors)))
        for name in sorted(tensors):
            a = tensors[name]
            a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
            a = np.ascontiguousarray(a, dtype=np.float16 if fp16 else np.float32)
            if a.ndim == 0:
                a = a.reshape(1)
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<BB", 1 if fp16 else 0, a.ndim))
            f.write(struct.pack(f"<{a.ndim}i", *a.shape))
            f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="a model name (%s) or a checkpoint path" % ", ".join(MODELS))
    ap.add_argument("out")
    ap.add_argument("--fp32", action="store_true", help="store float32 rather than float16")
    ap.add_argument("--prefix", default="",
                    help="keep only the tensors under this prefix, and strip it")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop the tensors under this prefix (repeatable)")
    a = ap.parse_args()
    tensors, folded = fold(load_state(a.source, a.prefix, a.exclude))
    if not tensors:
        sys.exit(f"nothing under prefix '{a.prefix}'")
    write_tgw(a.out, tensors, fp16=not a.fp32)
    n = sum(t.numel() for t in tensors.values())
    print(f"{a.out}: {len(tensors)} tensors, {n / 1e6:.1f}M values, "
          f"{folded} BatchNorms folded, {'float32' if a.fp32 else 'float16'}")


if __name__ == "__main__":
    main()
