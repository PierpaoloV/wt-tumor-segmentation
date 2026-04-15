# WT Tumour Segmentation

Pixel-level segmentation of Wilms' tumour (nephroblastoma) histopathology whole-slide images. This repository contains the training configuration, annotation rasterisation scripts, and data split utilities for retraining the [pathology-segmentation-pipeline](https://github.com/DIAGNijmegen/pathology-segmentation-pipeline) on a 61-patient cohort annotated at the Princess Máxima Center, Utrecht.

The work reproduces and extends the methodology of van Alst (2021), targeting the thesis best result of F1 = 0.850 achieved with a DenseNet-201 encoder.

---

## Repository structure

```
wt-tumor-segmentation/
├── src/wt_segmentation/
│   └── labels.py                  # Canonical Dutch → English → mask-int class dictionary
├── scripts/
│   ├── rasterise_annotations.py   # ASAP XML → TIFF masks at 2.0 µm
│   ├── make_splits.py             # Patient-level 70/15/15 train/val/test split
│   ├── train.sh                   # Docker wrapper for training
│   └── evaluate.sh                # Docker wrapper for evaluation
├── configs/
│   ├── wt_network_configuration.yaml   # Config B: U-Net + DenseNet-201
│   └── wt_albumentations.yaml          # Augmentation pipeline
├── data/                          # Generated YAML split files (not committed)
└── results/                       # Scores and WandB run IDs (no weights)
```

---

## Dependencies

All training and inference code lives in the **pathology-segmentation-pipeline** repository. It must be available as a Docker image named `pathology-pipeline` before running any of the scripts below.

```
https://github.com/DIAGNijmegen/pathology-segmentation-pipeline
```

Build the image once from the pipeline repository:

```bash
cd /path/to/pathology-segmentation-pipeline
docker build -t pathology-pipeline .
```

---

## Class scheme

16 classes, mask integer 0–15. Fibrose is merged into Regressie (class 6) at rasterisation. Unannotated pixels receive value 255 and are automatically ignored in the loss.

| Int | English | Dutch |
|-----|---------|-------|
| 0 | background | Achtergrond |
| 1 | wt_blastema | WT-blasteem |
| 2 | wt_stroma | WT-stroma |
| 3 | wt_epithelium | WT-epitheel |
| 4 | necrosis | Necrose |
| 5 | bleeding | Bloeding |
| 6 | regression (+ fibrosis) | Regressie + Fibrose |
| 7 | glomeruli | Glomeruli |
| 8 | tubules | Tubuli |
| 9 | nephrogenic_rest | Nefrogene rest |
| 10 | fat | Vet |
| 11 | connective_tissue | Bindweefsel |
| 12 | blood_vessels | Bloedvaten |
| 13 | nerves | Zenuwtakken |
| 14 | lymph_nodes | Lymfklier |
| 15 | urothelium | Urotheel |

The full mapping is defined in `src/wt_segmentation/labels.py` and imported by every script — editing that file is the only place you need to change class definitions.

---

## Step 1 — Generate masks

Rasterise the 108 root-level ASAP XML annotations to TIFF masks at 2.0 µm. Run inside the Docker image so ASAP and `pathology-common` are available.

```bash
docker run --rm \
    -v /path/to/annotations:/home/user/data/annotations:ro \
    -v /path/to/images:/home/user/data/images:ro \
    -v /path/to/masks:/home/user/data/masks \
    -v "$(pwd)/scripts:/home/user/project/scripts:ro" \
    -v "$(pwd)/src:/home/user/project/src:ro" \
    pathology-pipeline \
    python3 /home/user/project/scripts/rasterise_annotations.py \
        --xml_dir   /home/user/data/annotations \
        --image_dir /home/user/data/images \
        --out_dir   /home/user/data/masks \
        --spacing   2.0 \
        --workers   4
```

This writes one `<stem>.tif` mask and one `<stem>.json` sidecar per slide. The sidecar records which class integers are present in that slide — required for the next step.

---

## Step 2 — Generate data splits

Create the patient-level 70/15/15 train/val/test split and emit `data/wt_train.yaml` and `data/wt_test.yaml`.

```bash
python3 scripts/make_splits.py \
    --mask_dir  /path/to/masks \
    --image_dir /path/to/images \
    --out_dir   data
```

**After rasterisation**, regenerate the sampling weights from real pixel counts instead of the polygon-count approximation:

```bash
python3 scripts/make_splits.py \
    --mask_dir  /path/to/masks \
    --image_dir /path/to/images \
    --out_dir   data \
    --recompute_weights
```

Paste the printed `label_dist` block into `configs/wt_network_configuration.yaml`.

---

## Step 3 — Train

Set the two required environment variables and run `train.sh`. The script mounts your data and config directories into Docker and launches `pytorch_exp_run.py`.

```bash
export WT_DATA_ROOT=/path/to/data_root    # directory containing images/ and masks/
export WT_OUTPUT_ROOT=~/wt_runs           # host path for checkpoints and logs

bash scripts/train.sh wt_configB_run1
```

The training script requires an interactive terminal (`-it`) because `pytorch_exp_run.py` prompts for architecture selection at startup — select **unet** when prompted.

WandB API key is read from a `.env` file in the repo root if present:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

Trained weights are not stored in this repository. Upload them to HuggingFace after training:

```
huggingface-cli upload PierpaoloV93/pathology-segmentation-models \
    <checkpoint.pt> wt-segmentation/<run_name>.pt
```

---

## Step 4 — Evaluate

Run held-out evaluation on the test split. The predictions directory must already exist, produced by the pipeline's async tile inference engine.

```bash
export WT_DATA_ROOT=/path/to/data_root
export WT_OUTPUT_ROOT=~/wt_runs

bash scripts/evaluate.sh wt_configB_run1
```

Scores are written to `${WT_OUTPUT_ROOT}/wt_configB_run1/scores.yaml`. Target metrics:

| Metric | Target |
|--------|--------|
| Overall F1 | ≥ 0.80 |
| WT-blastema F1 | ≥ 0.70 |

---

## Reference

van Alst, L. (2021). *Deep Learning for Segmentation and Classification of Wilms' Tumours*. MSc Thesis, Radboud University / Radboudumc.
