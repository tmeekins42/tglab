#!/usr/bin/env python3
"""nn_reference -- run a published network in PyTorch and save what it
computes, for tglab's own implementation to be checked against.

Development-time only, like nn_convert.py. The output is a .tgw (float32)
holding the network's input, its intermediate feature maps and its result,
so a mismatch can be traced to the layer where it starts rather than only
seen at the end. tools/bench_nn.cpp reads it.

Runs in float32 with autocast OFF: the published code autocasts to bfloat16,
which would make the reference itself the noisy side of the comparison.

usage:
    python tools/nn_reference.py dad <image> out.tgw [--keypoints N] [--resize 1024]
    python tools/nn_reference.py dedode_b128 <image> out.tgw [--resize 784] [--light]
    python tools/nn_reference.py loma_b128 <image> <image2> out.tgw [--keypoints 2048]

Needs the LoMa package (pip install -e <LoMa checkout>) for the model code.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from nn_convert import write_tgw  # noqa: E402


def no_autocast(model):
    for m in model.modules():
        if hasattr(m, "amp"):
            m.amp = False


def dad(image, n, resize):
    from loma.detector.dad import DaD

    model = DaD(DaD.Cfg(compile=False, resize=resize)).eval()
    no_autocast(model)
    out = {}
    with torch.inference_mode():
        x = model.load_image(image, device="cpu")          # 1x3xHxW, 0..1
        out["image"] = x[0]
        xn = model.normalizer(x)
        out["input"] = xn[0]
        feats, sizes = model.encoder(xn)
        for f, s in zip(feats, (1, 2, 4, 8)):
            out[f"feat{s}"] = f[0]
        logits = xn[:, :1].new_zeros(())
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (fm, scale) in enumerate(zip(reversed(feats), scales)):
            delta, context = model.decoder(fm, context=context, scale=scale)
            out[f"delta{scale}"] = delta[0]
            out[f"context{scale}"] = context[0]
            logits = logits + delta.float()
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context.float(), size=size, mode="bilinear",
                                        align_corners=False)
        out["logits"] = logits[0]
        det = model(x, num_keypoints=n)
        _, _, H, W = x.shape
        kp = model.to_pixel_coords(det["keypoints"], H, W)
        out["keypoints"] = kp[0]
        out["keypoint_probs"] = det["keypoint_probs"][0]
    return out


def dedode_b128(image, n, size, light):
    """LoMa-B128's descriptor, on DaD's keypoints, the way LoMa runs it from a
    path: DaD at 1024 keeping the aspect ratio, the descriptor on the image
    squashed to size x size, sampled at the same normalised positions."""
    from loma.descriptor.dedode import DeDoDeDescriptor
    from loma.detector.dad import DaD

    url = "https://github.com/davnords/storage/releases/download/loma/loma_B128.pth"
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
    desc = DeDoDeDescriptor(DeDoDeDescriptor.Cfg(arch="dedode_b", compile=False,
                                                 descriptor_dim=128)).eval()
    missing, unexpected = desc.load_state_dict(
        {k[len("_descriptor."):]: v for k, v in sd.items() if k.startswith("_descriptor.")})
    assert not missing and not unexpected
    no_autocast(desc)
    det = DaD(DaD.Cfg(compile=False)).eval()
    no_autocast(det)

    out = {}
    with torch.inference_mode():
        kp = det.detect_from_path(image, num_keypoints=n)["keypoints"]   # 1xNx2, normalised
        out["kp_norm"] = kp[0]
        x = desc.read_image(image, H=size, W=size)                       # 1x3xHxW, 0..1
        out["input"] = x[0]
        if not light:
            feats, sizes = desc.encoder(x)
            for f, s in zip(feats, (1, 2, 4, 8)):
                out[f"feat{s}"] = f[0]
            context = None
            for scale in desc.decoder.scales:
                fm = feats[(1, 2, 4, 8).index(int(scale))]
                delta, context = desc.decoder(fm, scale=scale, context=context)
                out[f"delta{scale}"] = delta[0]
                if scale != "1":
                    nxt = feats[(1, 2, 4, 8).index(int(scale)) - 1]
                    context = F.interpolate(context, size=nxt.shape[-2:], mode="bilinear",
                                            align_corners=False)
            out["descmap"] = desc(x)[0]
        out["descriptors"] = desc.describe_keypoints(x, kp)["descriptions"][0]
    return out


def loma_b128(image, image2, n):
    """LoMa-B128's matcher on a pair, on its own keypoints and descriptors:
    each layer's descriptors for both images, the final similarity, and the
    matches filter_matches keeps."""
    from loma.loma import LoMa, LoMaB128, filter_matches

    model = LoMa(LoMaB128(mp=False)).eval()
    no_autocast(model)
    out = {}
    with torch.inference_mode():
        k0, d0, _, _ = model.detect_and_describe(image, n)
        k1, d1, _, _ = model.detect_and_describe(image2, n)
        out["kpts0"], out["kpts1"] = k0[0], k1[0]        # normalised, -1..1
        out["desc0"], out["desc1"] = d0[0], d1[0]        # raw, 128 floats
        x0, x1 = model.input_proj(d0), model.input_proj(d1)
        out["proj0"], out["proj1"] = x0[0], x1[0]
        e0, e1 = model.posenc(k0), model.posenc(k1)
        for i in range(model.cfg.n_layers):
            x0, x1 = model.transformers[i](x0, x1, e0, e1)
            out[f"layer{i}_0"], out[f"layer{i}_1"] = x0[0], x1[0]
        scores, sim = model.log_assignment[model.cfg.n_layers - 1](x0, x1)
        out["sim"] = sim[0]
        m0, _, ms0, _ = filter_matches(scores, model.cfg.filter_threshold)
        out["m0"] = m0[0].float()                        # match in image 1, or -1
        out["mscores0"] = ms0[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("network", choices=["dad", "dedode_b128", "loma_b128"])
    ap.add_argument("image")
    ap.add_argument("image2", nargs="?", help="loma_b128: the second image of the pair")
    ap.add_argument("out")
    ap.add_argument("--keypoints", type=int, default=4096)
    ap.add_argument("--resize", type=int, default=None,
                    help="dad: the long side (1024); dedode_b128: the square side (784)")
    ap.add_argument("--light", action="store_true",
                    help="dedode_b128: skip the per-layer maps, keep the descriptors")
    a = ap.parse_args()
    if a.network == "dad":
        out = dad(a.image, a.keypoints, a.resize or 1024)
    elif a.network == "loma_b128":
        out = loma_b128(a.image, a.image2, a.keypoints)
    else:
        out = dedode_b128(a.image, a.keypoints, a.resize or 784, a.light)
    write_tgw(a.out, out, fp16=False)
    for k, v in out.items():
        print(f"  {k:16s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
