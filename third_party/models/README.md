# Network weights

The learned feature stages read their weights from `models/*.tgw` beside the
executable. The files are not tracked here; they are generated from the
published checkpoints by `tools/nn_convert.py`:

| File | Used by | Source | Licence |
|---|---|---|---|
| `models/dad.tgw` | `detect_dad` | DaD, Edstedt et al. 2025 — <https://github.com/Parskatt/dad> | MIT ([DaD-LICENSE.txt](DaD-LICENSE.txt)) |
| `models/dedode_b128.tgw` | `describe_dedode` | LoMa-B128's DeDoDe-B descriptor, Edstedt, Nordström et al. 2026 — <https://github.com/davnords/LoMa> | MIT ([LoMa-LICENSE.txt](LoMa-LICENSE.txt)) |
| `models/loma_b128.tgw` | `match_ann(method = 1)` | LoMa-B128's matcher — same repository | Apache-2.0, inherited from LightGlue ([LightGlue-LICENSE.txt](LightGlue-LICENSE.txt)) |

```sh
python tools/nn_convert.py dad models/dad.tgw
python tools/nn_convert.py loma_b128 models/dedode_b128.tgw --prefix _descriptor.
python tools/nn_convert.py loma_b128 models/loma_b128.tgw --exclude _detector. --exclude _descriptor.
```

The release archive ships all three with these licence texts.
