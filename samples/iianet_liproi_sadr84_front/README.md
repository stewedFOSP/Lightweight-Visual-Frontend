# IIANet liproi32x32 + sadr84-front sample inputs

Preprocessed tensors for demo clips that match
`docs/assets/audio/proposed/sample1_s1.wav` and `sample2_s2.wav`.

| Sample | LRS3 tt utterance | Source speaker wav |
| --- | --- | --- |
| `sample1_s1` | `0VJqrlH9cdI_00002_0.49459_t7Xr3AsBEK4_00006_-0.49459` | `.../tt/s1/<utt>.wav` |
| `sample2_s2` | `0ZfSOArXbGQ_00003_1.188_RplnSVTzvnU_00003_-1.188` | `.../tt/s2/<utt>.wav` |

Each folder contains:

- `mix.wav`: mixture at 16 kHz
- `ref.wav`: clean target speaker
- `liproi32x32.npz`: key `data`, shape `[T, 32, 32]` (test crop + normalize)
- `lmk84.npz`: key `data`, shape `[T, 84, 3]` (sadr84-front, mean/std normalized)
- `meta.txt`: source paths / shapes
