# MoCos PyTorch Usage

This file documents the PyTorch reimplementation entry point, `MoCos_torch.py`.
It is intended for users who do not want to install the original TensorFlow 1.x
environment.


## What Changed

`MoCos_torch.py` removes the TensorFlow 1.14 runtime dependency and implements
the main MoCos training and evaluation flow in PyTorch:

- Motif guided graph transformer (MGT)
- Laplacian skeleton positional encoding
- Sub-skeleton CSP loss (SSk-CSP)
- Sub-tracklet CSP loss (STr-CSP)
- Probe/gallery re-ID evaluation with mAP, Rank-1, Rank-5, Rank-10

The PyTorch script reuses the original preprocessed `.npy` data format and the
existing data loading utilities in `utils/`.

Important: the author's released pretrained models are TensorFlow `.ckpt`
checkpoints. They cannot be loaded directly by `MoCos_torch.py`. Use `MoCos.py`
with a TensorFlow 1.x environment for those checkpoints, or retrain PyTorch
weights with `MoCos_torch.py`.

## Environment

Create a fresh environment:

```bash
conda create -n mocos-torch python=3.10 -y
conda activate mocos-torch
```

Install PyTorch first.

For CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Verify the script starts:

```bash
python MoCos_torch.py --help
```

## Data

Use the same preprocessed data layout as the original project.

Place preprocessed datasets under `Datasets/`:

```text
MoCos/
├─ MoCos_torch.py
├─ Datasets/
│  ├─ KS20/
│  │  └─ 6/
│  │     ├─ train_npy_data/
│  │     └─ test_npy_data/
│  │        ├─ gallery/
│  │        └─ probe/
│  ├─ KGBD/
│  ├─ IAS/
│  └─ BIWI/
└─ ReID_Models/
```

If you need to generate `.npy` files from the original raw datasets, use the
original preprocessing script:

```bash
python Data-process.py 6
```

The original README explains the required raw dataset folders in more detail.

## Quick Smoke Test

Once data exists, run a short training job to confirm the full pipeline works:

```bash
python MoCos_torch.py --dataset KS20 --probe probe --length 6 --epochs 5 --patience 5 --device cpu
```

This is only a sanity check. It is not expected to produce a good model.

## Training

Recommended starting command:

```bash
python MoCos_torch.py --dataset KS20 --probe probe --length 6 --epochs 1000 --patience 80 --save_model 1
```

Other dataset examples:

```bash
python MoCos_torch.py --dataset KGBD --probe probe --length 6 --epochs 1000 --patience 80 --save_model 1
python MoCos_torch.py --dataset IAS --probe A --length 6 --epochs 1000 --patience 80 --save_model 1
python MoCos_torch.py --dataset IAS --probe B --length 6 --epochs 1000 --patience 80 --save_model 1
python MoCos_torch.py --dataset BIWI --probe Walking --length 6 --epochs 1000 --patience 80 --save_model 1
python MoCos_torch.py --dataset BIWI --probe Still --length 6 --epochs 1000 --patience 80 --save_model 1
```

CASIA-B example:

```bash
python MoCos_torch.py --dataset CASIA_B --probe_type nm.nm --length 40 --epochs 1000 --patience 80 --save_model 1
```

`--epochs` is a maximum epoch count. Training stops earlier when Rank-1 does not
improve for `--patience` epochs.

You can interrupt training with `Ctrl+C`. The current script saves only the best
model when `--save_model 1` is enabled and Rank-1 improves. It does not yet save
full resumable optimizer checkpoints.

## Evaluation

Evaluate a PyTorch checkpoint produced by `MoCos_torch.py`:

```bash
python MoCos_torch.py --dataset KS20 --probe probe --length 6 --mode Eval
```

The checkpoint path is generated from the dataset, probe, sequence length, and
training hyperparameters. For example:

```text
ReID_Models/KS20/probe/_MoCos_Torch_f_6_prob_s_0.25_prob_t_0.25_lambda_0.9/best.pt
```

For CASIA-B:

```bash
python MoCos_torch.py --dataset CASIA_B --probe_type nm.nm --length 40 --mode Eval
```

The script prints:

```text
mAP | R1 | R5 | R10
```

## Using the Author's Pretrained Models

The pretrained models linked in the original README are TensorFlow checkpoints.
Run them with the original script:

```bash
python MoCos.py --dataset KS20 --probe probe --length 6 --mode Eval
```

That path requires the old environment:

```text
Python 3.7
TensorFlow 1.14
PyTorch 1.1
CUDA 9.x / cuDNN 7.x
```

`MoCos_torch.py` cannot directly load those `.ckpt` files.

## Main Arguments

Common arguments:

```text
--dataset          IAS, KGBD, KS20, BIWI, CASIA_B
--probe            probe, A, B, Walking, Still
--probe_type       CASIA-B setting, e.g. nm.nm, cl.cl, bg.bg, cl.nm, bg.nm
--length           sequence length
--batch_size       batch size, default 256
--epochs           maximum training epochs
--patience         early stopping patience
--save_model       set to 1 to save the best PyTorch checkpoint
--device           cpu, cuda:0, cuda:1, etc.
```

Model arguments:

```text
--H                hidden size, default 128
--n_heads          number of attention heads, default 8
--L_transformer    number of MGT layers, default 2
--enc_k            Laplacian positional encoding dimension, default 10
--prob_s           spatial node masking probability
--prob_t           temporal frame masking probability
--fusion_lambda    SSk/STr CSP fusion weight
--t_1              STr-CSP temperature
--t_2              SSk-CSP temperature
```

Several dataset-specific defaults are applied inside `MoCos_torch.py` to match
the original code more closely.

## Notes

- `requirements-torch.txt` is for the PyTorch entry point.
- Use `MoCos_torch.py` for new training in a modern PyTorch environment.
