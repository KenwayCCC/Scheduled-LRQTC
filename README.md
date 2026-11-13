# Scheduled-LRQTC

Official implementation of **Scheduled-LRQTC** for color video completion. The method combines low-rank quaternion tensor modeling with complementary spatial and spatiotemporal denoising priors. A dynamic probability schedule coordinates the two priors during reconstruction.

## Repository layout

```text
Scheduled-LRQTC/
├── main.py                # Reconstruction entry point
├── utils/                 # Reconstruction and denoising utilities
├── data/                  # Video sequences and observation masks
├── ffdnet/                # FFDNet architecture and checkpoint
├── fastdvdnet_model/      # FastDVDnet architecture and checkpoint
└── requirements.txt       # Python dependencies
```

## Environment

The implementation requires **64-bit Python 3.10 on Windows** and a CUDA-capable NVIDIA GPU. Create and activate a Python environment, install a CUDA-compatible PyTorch build following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), and install the remaining dependencies:

```powershell
conda create -n scheduled-lrqtc python=3.10
conda activate scheduled-lrqtc
python -m pip install -r requirements.txt
```

## Quick start

Reconstruct the bundled Foreman sequence with the random mask at a missing rate of 0.95:

```powershell
python main.py --dataset foreman --mask data/masks/observed_mask_mr95_seed0.npy --output output/foreman_mr95.npy
```

The available sequences are `akiyo`, `carphone`, `container`, `foreman`, `grandma`, `hall`, `news`, and `suzie`. Bundled random and structured masks are located in `data/masks/` and `data/masks/structured/`, respectively.

To use a custom video and mask:

```powershell
python main.py --input path/to/video.npy --mask path/to/mask.npy --output output/recovered.npy
```

The input video must be a `uint8` NumPy array with shape `[T, 256, 256, 3]` in RGB order. The mask must have shape `[T, 256, 256]`, with `1` for observed pixels and `0` for missing pixels. The output is a `float32` NumPy array in normalized RGB units with shape `[T, 256, 256, 3]`.

## Command-line interface

```powershell
python main.py --help
```

| Option | Description |
| --- | --- |
| `--dataset` | Select a bundled video sequence. |
| `--input`, `--mask` | Provide a video array and an observation mask. `--input` overrides `--dataset`. |
| `--output` | Set the path for the reconstructed video array. |
| `--device` | Select the computation device, such as `cuda:0`. |
| `--K` | Set the maximum iteration index. The algorithm runs for `k = 0, ..., K`. |
| `--eta` | Set one shared low-rank weight or one weight per tensor unfolding. |
| `--lambda` | Set the PnP regularization weight. |
| `--mu`, `--rho` | Set the initial penalty and its growth factor. |
| `--gamma`, `--epsilon` | Set the parameters for singular value reweighting. |
| `--p-min`, `--p-max` | Set the endpoints of the denoiser selection schedule. |
| `--seed` | Set the random seed. |

## Reproducibility

The default random seed is `0`. For controlled comparisons, use the same video, mask, parameters, and seed across runs. The default parameter values provide a starting configuration; use the experiment-specific settings when reproducing reported results.

## Citation

If you find this repository useful, please cite the accompanying Scheduled-LRQTC paper. Citation information will be added when the manuscript is publicly available.

## Acknowledgements

This repository includes pretrained denoising components based on FFDNet and FastDVDnet. Please refer to the corresponding original projects and publications when using these components.
