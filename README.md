# CGMatch: A Different Perspective of Semi-supervised Learning

**Reproduction project** for the CVPR 2025 paper by Bo Cheng, Jueqing Lu, Yuan Tian, Haifeng Zhao, Yi Chang, and Lan Du.

- **Paper:** [arXiv:2503.02231](https://arxiv.org/abs/2503.02231)
- **Official repo:** [BoCheng-96/CGMatch](https://github.com/BoCheng-96/CGMatch)
- **Base framework:** [Microsoft USB](https://github.com/microsoft/Semi-supervised-learning)
- **Our fork:** [Alam8010/CGMatch](https://github.com/Alam8010/CGMatch)

---

## What CGMatch Does (Plain English)

Standard semi-supervised learning methods like FixMatch decide whether to trust a model's prediction on an unlabeled image based only on **confidence** — how sure the model is right now. This works poorly when labels are very scarce (e.g., only 4 per class), because an undertrained model produces unreliable confidence scores.

CGMatch adds a second signal called **Count-Gap (CG)**: how *consistent* the model's prediction has been across recent training steps. Using both confidence and consistency together, CGMatch sorts every unlabeled image into one of three groups:

| Group | Condition | Treatment |
|-------|-----------|-----------|
| Easy | High confidence + consistent | Hard label, standard cross-entropy loss |
| Ambiguous | Medium confidence or inconsistent | Soft label, GCE loss (more forgiving) |
| Hard | Low confidence + inconsistent | Skipped |

The key claim: Count-Gap makes pseudo-labels more trustworthy when labels are scarce, leading to better training signal.

---

## Ablation Study

**Question:** Does the Count-Gap metric specifically drive CGMatch's performance, or does any sorting of unlabeled data into subsets achieve a similar result?

We trained three variants on CIFAR-10 with the 40-label setting (4 labeled images per class):

| Variant | Description | Accuracy | Error Rate |
|---------|-------------|----------|------------|
| CGMatch (full) | Count-Gap guides easy/ambiguous/hard sorting | 20.26% | 79.74% |
| CGMatch (random sorting) | Random assignment to subsets, no Count-Gap | 16.79% | 83.21% |
| FixMatch (baseline) | Confidence only, no subset sorting | 14.56% | 85.44% |

**Finding:** Count-Gap provides a meaningful improvement over both random sorting and plain FixMatch. Random sorting itself still outperforms FixMatch, suggesting the three-loss structure helps — but Count-Gap is what makes the sorting meaningful. This matches the paper's findings directionally.

### Why our numbers differ from the paper

The paper reports CGMatch at **4.87% error** and FixMatch at **8.33%** on the same setting. Our numbers are much higher because:

- Paper trained for **1,048,576 iterations**; we trained for **1,024 iterations** due to free compute limits on Kaggle
- We used **T4 x2 GPU** on Kaggle free tier
- With less than 0.1% of the paper's training budget, the model does not fully converge — it is still in early learning stages when training stops
- Despite the absolute accuracy gap, the **relative ordering** of the three variants matches the paper exactly: CGMatch > random sorting > FixMatch

---

## Compatibility Patches

The original CGMatch code was written for Python 3.8–3.10. Running it on Kaggle's Python 3.12 environment required the following fixes, all of which are applied in this fork:

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `semilearn/nets/__init__.py` | `timm.models.layers.helpers` removed in newer timm | Commented out ViT import |
| 2 | `semilearn/core/hooks/__init__.py` | `AimHook` import error | Commented out AimHook import |
| 3 | `semilearn/core/algorithmbase.py` | Empty `if self.args.use_aim:` block after removing AimHook | Added `pass` statement |
| 4 | `semilearn/core/utils/misc.py` | `ruamel.yaml` API changed | Replaced with `import yaml` |
| 5 | `semilearn/algorithms/cgmatch/cgmatch.py` | `y_true[i].item()` fails on non-scalar tensor | Changed to `int(y_true.flatten()[i])` |
| 6 | `semilearn/algorithms/cgmatch/cgmatch.py` | ECE tensor shape mismatch `(448,10)` vs `(448,)` | Added `argmax(dim=1)` when `dim > 1` |
| 7 | `config/classic_cv/cgmatch/cgmatch_cifar10_40_0.yaml` | `gpu: None` overriding command line | Set `gpu: 0` directly in YAML |
| 8 | `semilearn/core/hooks/evaluation.py` | `algorithm.warm_up_iter` missing on FixMatch | Wrapped in `hasattr()` check |
| 9 | `semilearn/core/hooks/logging.py` | Same `warm_up_iter` issue in logging hook | Same `hasattr()` fix |
| 10 | `semilearn/algorithms/cgmatch/cgmatch.py` | Random sorting variant | Replaced count_gap condition with `random.random() < 0.5` |

---

## Setup and Training

### Requirements

- Kaggle account with GPU enabled (T4 or P100)
- Internet turned on in Kaggle notebook settings (Settings → Internet → On)
- Python 3.12 (Kaggle default)

### Step 1 — Run the master setup cell

```python
import subprocess, os, shutil

USB_PATH = '/kaggle/working/USB'
CG_PATH  = '/kaggle/working/CGMatch'

os.system(f"rm -rf {USB_PATH} {CG_PATH}")
os.system(f"git clone https://github.com/microsoft/Semi-supervised-learning {USB_PATH}")
os.system(f"git clone https://github.com/Alam8010/CGMatch.git {CG_PATH}")

shutil.copytree(f'{CG_PATH}/semilearn', f'{USB_PATH}/semilearn', dirs_exist_ok=True)
shutil.copytree(f'{CG_PATH}/config',    f'{USB_PATH}/config',    dirs_exist_ok=True)
shutil.copy(f'{CG_PATH}/train.py', f'{USB_PATH}/train.py')

os.system("pip install timm==0.6.12 progress ruamel.yaml tensorboard tqdm scikit-learn -q")

r = subprocess.run(
    ['python', '-c', 'from semilearn.algorithms import get_algorithm; print("Import OK")'],
    cwd=USB_PATH, capture_output=True, text=True
)
print(r.stdout.strip() or r.stderr.strip())
```

Expected output: `Import OK`

All patches are already applied in this fork and get copied automatically via the `copytree` step. No manual patching needed.

### Step 2 — Train CGMatch

```bash
cd /kaggle/working/USB && python train.py \
    --c config/classic_cv/cgmatch/cgmatch_cifar10_40_0.yaml \
    --save_dir /kaggle/working/results
```

### Step 3 — Train FixMatch

```bash
cd /kaggle/working/USB && python train.py \
    --c config/classic_cv/fixmatch/fixmatch_cifar10_40_0.yaml \
    --save_dir /kaggle/working/results
```

### Step 4 — Train CGMatch with random sorting

The random sorting variant is already patched in this fork — `cgmatch.py` replaces the Count-Gap condition with `random.random() < 0.5`. Run with the same CGMatch config.

---

## Dataset

**CIFAR-10** — 60,000 images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), each 32×32 pixels. Downloads automatically via `torchvision.datasets.CIFAR10()`. No manual download needed.

**40-label setting:** Only 40 images have labels (4 per class). The remaining 49,960 are used as unlabeled data. This is the most challenging setting in the paper.

---

## Deployment

A Streamlit app loads the trained CGMatch model and allows users to upload any image to receive:
- Predicted class label (one of the 10 CIFAR-10 classes)
- Top-3 class probabilities
- Whether the model's prediction falls into the easy, ambiguous, or hard category

The trained model checkpoint (`model_best.pth`) is available separately — file size (592MB) exceeds GitHub's limit.

---

## Team

- Akash Alam (ERP: 31682)
- Mubeen Udin (ERP: 30909)

Introduction to Machine Learning — Semester Project
