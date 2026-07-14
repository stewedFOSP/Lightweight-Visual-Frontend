# A Lightweight Dual-Branch Visual Front-End for Efficient Audio-Visual Target Speaker Extraction

Anonymous companion repository for the paper demo and review materials.

## Paper

**A Lightweight Dual-Branch Visual Front-End for Efficient Audio-Visual Target Speaker Extraction**

## Anonymous demo (GitHub Pages)

Anonymous mirror / Pages URL (replace with the URL returned by [anonymous.4open.science](https://anonymous.4open.science/) after anonymization):

```text
https://anonymous.4open.science/r/<ANON-REPO-ID>/
```

Once GitHub Pages is enabled on the anonymized repository (`Settings → Pages → Deploy from a branch → main /docs`), the static demo will be served from:

```text
https://anonymous.4open.science/w/<ANON-REPO-ID>/
```

> Keep this README free of author names, affiliations, and non-anonymous personal links for blind review.

## What the demo shows

The site in [`docs/`](docs/) includes:

1. **Spatial attention videos** — four attention heads from the lightweight visual front-end  
2. **Audio comparison table** (layout inspired by [this AVSE demo](https://helpful-rabanadas-47fa43.netlify.app/))  
   - **Mix**: 1 row × 2 clips (Sample 1 / Sample 2)  
   - **Target**: 1 row × 4 clips (each sample’s clean s1 / s2)  
   - **ResNet18 (pretrained)**, **BlazeNet64**, **Proposed (32×32 + lmk84)**: each 1 row × 4 clips  
   - Training setting for compared models: `tolerance=5`, `early_stop=15`  
   - Every entry has a spectrogram image + audio player (placeholders until you replace assets)

## Asset layout

| Type | Path | Filenames |
|------|------|-----------|
| Attention videos | `docs/assets/videos/` | `S50001_mask{0-3}.mp4` |
| Spectrograms | `docs/assets/picture/{mix,target,resnet18,blazenet64,proposed}/` | `sample*.svg` (swap for PNG/JPG as needed) |
| Audio | `docs/assets/audio/{mix,target,resnet18,blazenet64,proposed}/` | see `docs/assets/audio/README.md` |

Replace placeholder spectrograms and missing `.wav` files in place; paths in `docs/index.html` stay the same if filenames are kept.

## Enable GitHub Pages

Static site lives in `docs/` so it can be published from the same branch as the anonymous mirror (`anonymous.4open.science` only supports Pages content that lives in the mirrored branch).

1. Open **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **main**, folder: **/docs**

Non-anonymous local preview URL (not for review):

`https://stewedfosp.github.io/Lightweight-Visual-Frontend/`
