# MoCos-torch Handoff

This project is the standalone PyTorch migration of the original MoCos
TensorFlow 1.x repository.

Current working directory:

```text
D:\Study\project\ReID\MoCos-torch
```

## Current Files

Main files:

```text
MoCos_torch.py        PyTorch training/evaluation entry point
README.md            PyTorch usage notes
requirements.txt     Modern PyTorch dependency list
Data-process.py      Original data preprocessing script
utils/               Original data loading utilities
tools/               Extra conversion utilities
Datasets/            Local preprocessed data
ReID_Models/         Local checkpoints
```

## Important Recent Fixes

The following fixes have been applied to `MoCos_torch.py`:

1. Dataset defaults no longer overwrite explicitly passed CLI arguments.

   These arguments now stay user-controllable:

   ```text
   --prob_s
   --prob_t
   --fusion_lambda
   --rand_flip
   ```

2. Training loop order was changed.

   Old behavior:

   ```text
   extract prototypes -> evaluate -> early stop/save -> train
   ```

   New behavior:

   ```text
   extract prototypes -> train -> evaluate -> save best -> early stop
   ```

3. BIWI / 20-joint motif graphs were made closer to the original TensorFlow
   code.

   Specifically, `adj4` and `adj5` now use the original-style hand/arm and
   foot/leg collaborative motif definitions instead of simple cliques.

4. Extra self-loops were removed from motif adjacency matrices.

5. `motif_adjs` is now registered as a non-persistent buffer:

   ```python
   self.register_buffer("motif_adjs", ..., persistent=False)
   ```

   This prevents old checkpoints from overwriting the fixed motif graph.

6. Evaluation checkpoint loading now uses:

   ```python
   load_torch_checkpoint(...)
   model.load_state_dict(..., strict=False)
   ```

   This reduces compatibility issues with earlier `.pt` checkpoints and avoids
   the PyTorch `torch.load` warning on newer versions.

7. Optional reproducibility controls were added:

   ```text
   --seed
   --deterministic
   ```

   `--seed` seeds Python, NumPy, and PyTorch. `--deterministic 1` requests
   deterministic PyTorch/CuDNN behavior when available.

8. Resumable training checkpoints were added.

   When `--save_model 1` is enabled, training now saves:

   ```text
   best.pt    best Rank-1 model with optimizer/epoch/best metrics/patience
   last.pt    latest epoch checkpoint with optimizer/epoch/best metrics/patience
   ```

   Resume with:

   ```bash
   python MoCos_torch.py ... --save_model 1 --resume auto
   ```

   Or pass a concrete `.pt` path to `--resume`.

9. Feature extraction now processes the final partial batch.

   Earlier PyTorch code used `range(0, len(data) - batch_size + 1, batch_size)`,
   which dropped any tail samples during prototype extraction and evaluation.
   Training batches now also include the final partial batch.

10. `--min_epochs` was added.

   Early stopping still uses `--patience`, but it will not stop before
   `epoch + 1 >= min_epochs`.

11. A TensorFlow-style initialization/BatchNorm experiment was reverted.

   Trying `Normal(0, 1)` for MGT Q/K/V and SSk-CSP projections plus TF-like BN
   settings made BIWI Walking worse in practice:

   ```text
   best mAP/R1: 0.2226/0.2895
   ```

   Keep PyTorch default Linear initialization and BatchNorm for now. If revisiting
   initialization, prefer a smaller controlled scale such as Xavier or
   `Normal(0, 1/sqrt(H))`, not raw `Normal(0, 1)`.

## Verification Already Run

From `D:\Study\project\ReID\MoCos-torch`:

```bash
python MoCos_torch.py --help
python -B -c "import ast, pathlib; ast.parse(pathlib.Path('MoCos_torch.py').read_text(encoding='utf-8')); print('syntax ok')"
```

Both passed.

Note: `python -m py_compile MoCos_torch.py` failed with:

```text
[WinError 5] access denied: '__pycache__'
```

That appears to be a local `__pycache__` permission issue, not a syntax issue.

## Recommended Training Commands

BIWI Walking:

```bash
python MoCos_torch.py --dataset BIWI --probe Walking --length 6 --epochs 800 --patience 150 --save_model 1 --gpu 0
```

Longer BIWI Walking run:

```bash
python MoCos_torch.py --dataset BIWI --probe Walking --length 6 --epochs 1500 --min_epochs 300 --patience 250 --save_model 1 --gpu 0
```

Evaluation:

```bash
python MoCos_torch.py --dataset BIWI --probe Walking --length 6 --mode Eval --gpu 0
```

Quick startup check:

```bash
python MoCos_torch.py --help
```

## Current Known Issue

The PyTorch implementation is a functional migration, not a strict variable-level
clone of the TensorFlow 1.x graph. It may not reproduce the paper or the
author's pretrained TensorFlow `.ckpt` results exactly.

The user observed before the latest fixes:

```text
[Epoch 130] mAP: 0.2717 | R1: 0.3229 | R5: 0.4440 | R10: 0.5000
Eval: mAP: 0.3260 | R1: 0.3841 | R5: 0.5078 | R10: 0.5664
```

That run used the earlier PyTorch code before the motif/default/training-loop
fixes. Retrain from scratch before judging the current version.

## Next Debug Priorities

If retraining is still weak, check these in order:

1. Compare PyTorch motif adjacency matrices against the original `MoCos.py`
   hardcoded matrices for each dataset, especially KS20 and CASIA_B.

2. Compare the TensorFlow and PyTorch attention behavior.

   The original multiplies attention logits by `adj_*`, which zeros non-motif
   logits before softmax instead of masking them to `-inf`. The PyTorch version
   follows that behavior.

3. Check pooling and CSP loss scaling.

   The original `SSk-CSP` sums frame CE losses then averages by batch. The
   PyTorch version follows that shape, but exact initialization and BN behavior
   differ.

4. Add reproducibility controls:
   Done at the CLI level. If strict experiment reproducibility is required,
   still document CUDA, PyTorch, driver, and dataset preprocessing versions.

5. Consider reducing evaluation overhead.

   Current training extracts full train/gallery/probe features each epoch. This
   follows the original high-level flow but can be slow.

## Notes for New Assistant Window

Use this directory as the working directory:

```text
D:\Study\project\ReID\MoCos-torch
```

Do not edit the old source directory unless the user explicitly asks:

```text
D:\Study\project\MoCos
```

The user's main goal is to keep the standalone PyTorch version usable and improve
training/evaluation quality without returning to the TensorFlow 1.x environment.
