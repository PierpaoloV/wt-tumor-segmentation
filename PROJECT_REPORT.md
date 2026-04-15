# Wilms' Tumour Segmentation — Project Report

**Date:** 2026-04-15  
**Goal:** Retrain the `pathology-segmentation-pipeline` on Wilms' tumour (WT) histopathology data, following the methodology of van Alst (2021).

---

## 1. Clinical Background

Wilms' tumour (nephroblastoma) accounts for ~90% of all childhood renal tumours. Risk stratification determines chemotherapy protocol:

| Risk | Dominant component |
|------|--------------------|
| Intermediate | Regressive, epithelial, stromal, mixed |
| High | Blastemal |

Automating pixel-level segmentation of WT tissue components reduces observer variability and pathologist workload.

---

## 2. Dataset

Images and masks are on a Samba share. Annotations (ASAP XML format) are taken exclusively from the root of `annotations/` — subfolders excluded.

### 2.1 Inventory

| | |
|---|---|
| XML files at root | 109 |
| Usable annotation files | **108** (1 duplicate excluded: `P000001_B107_dense_example_thomas`) |
| Unique patients | **61** |
| Slides per patient | 1–6 (median 2) |

**Distribution:** 34 patients have 1 slide, 18 have 2, 5 have 3, 4 have more.

> The `annotations/` subfolders (`all_dense/`, `dense_sophie/`, `tumor_thomas/`, etc.) contain 135 additional XML files covering 16 extra patients. Excluded for now; available as expansion data.

### 2.2 Annotated Classes

All labels are in Dutch. Scanning all 108 XMLs for actual drawn polygons (not ASAP group template definitions) yields **20 classes with real annotations**:

| English | Dutch | Slides | Polygons |
|---------|-------|--------|----------|
| Blood vessels | Bloedvaten | 79 | 944 |
| Connective tissue | Bindweefsel | 76 | 556 |
| Regression | Regressie | 61 | 484 |
| Bleeding | Bloeding | 59 | 251 |
| WT-Epithelium | WT-epitheel | 52 | 704 |
| WT-Stroma | WT-stroma | 50 | 349 |
| Fat | Vet | 50 | 298 |
| WT-Blastema | WT-blasteem | 47 | 441 |
| Glomeruli | Glomeruli | 46 | 557 |
| Tubules | Tubuli | 43 | 515 |
| Background | Achtergrond | 31 | 90 |
| Nerves | Zenuwtakken | 29 | 154 |
| Necrosis | Necrose | 24 | 229 |
| Urothelium | Urotheel | 13 | 73 |
| Nephrogenic rests | Nefrogene rest | 11 | 151 |
| Lymph nodes | Lymfklier | 7 | 58 |
| Fibrosis | Fibrose | 5 | 16 |
| Normal kidney tissue | Normaal nierweefsel | 1 | 1 |
| Adrenal cortex | Bijnierschors | 1 | 10 |
| Adrenal medulla | Bijniermerg | 1 | 7 |

### 2.3 Class Consolidation (following thesis, Chapter 4 pp. 36–38)

The thesis reduces 20 raw labels to **15 training classes**:

| Action | Label(s) | Reason |
|--------|----------|--------|
| Drop | Anaplasie | 0 annotations; not enough data |
| Drop | Bijnierschors, Bijniermerg | Not relevant to WT diagnosis; negligible count |
| Drop | Achtergrond | Handled by a separate tissue/background pre-processing network |
| Drop | Normaal nierweefsel | 1 annotation; not in thesis tables |
| Merge → Regression | Fibrose | Biologically similar to regressive tissue; only 16 annotations |

**Final 15 classes:**

| # | English | Dutch XML label | Category |
|---|---------|-----------------|----------|
| 1 | WT-blastema | WT-blasteem | WT tumour |
| 2 | WT-stroma | WT-stroma | WT tumour |
| 3 | WT-epithelium | WT-epitheel | WT tumour |
| 4 | Necrosis | Necrose | Chemo effect |
| 5 | Bleeding | Bloeding | Chemo effect |
| 6 | Regression (incl. Fibrosis) | Regressie + Fibrose | Chemo effect |
| 7 | Glomeruli | Glomeruli | Normal kidney |
| 8 | Tubules | Tubuli | Normal kidney |
| 9 | Nephrogenic rests | Nefrogene rest | Normal kidney |
| 10 | Fat | Vet | Normal stroma |
| 11 | Connective tissue | Bindweefsel | Normal stroma |
| 12 | Blood vessels | Bloedvaten | Normal stroma |
| 13 | Nerves | Zenuwtakken | Normal stroma |
| 14 | Lymph nodes | Lymfklier | Normal stroma |
| 15 | Urothelium | Urotheel | Normal stroma |

### 2.4 Data Split

No pre-existing split files found. A **patient-level** 70/15/15 split (train/val/test) is required to prevent data leakage across slides from the same patient — matching the thesis approach.

---

## 3. Thesis Baseline Results

The thesis trained on the same annotations using two architectures (15 classes, 0.5 µm pixel spacing):

| Model | Config | Annotations | F1 |
|-------|--------|-------------|----|
| U-Net (128×128, ~13M params) | Default | Sparse | 0.787 |
| U-Net (128×128, ~13M params) | Default | Sparse + dense | 0.777 |
| DenseNet-201 (412×412, ~18M params) | Default | Sparse | 0.801 |
| **DenseNet-201** | **Best** | **Sparse + dense** | **0.850** |
| DenseNet-201 (tumour/non-tumour post-processing) | Post-processed | Sparse + dense | 0.863 |

Both models used RMSProp, cross-entropy loss, ImageNet pre-training, LR decay on plateau (halve after 5 stagnant epochs), spatial/noise/colour augmentation.

---

## 4. Retraining Plan

### 4.1 Codebase: `pathology-segmentation-pipeline` (main branch)

Key capabilities relevant to this project:

- U-Net, UNet++, FPN, DeepLabV3+ via `segmentation-models-pytorch`; DenseNet-201 available as encoder
- Lovász / Dice / cross-entropy loss; ReduceLROnPlateau; AMP; early stopping; WandB
- Configurable patch size, pixel spacing, per-class label distribution weights
- Albumentations augmentation (HED colour normalisation, spatial, noise)
- Async tile inference engine for full WSI prediction output
- Docker (CUDA 12.4 / Python 3.10)

### 4.2 Proposed Configurations

| # | Architecture | Backbone | Patch | Spacing | Loss | Purpose |
|---|--------------|----------|-------|---------|------|---------|
| A | U-Net | EfficientNet-B0 | 512×512 | 0.5 µm | Lovász | Modern backbone baseline |
| **B** | **U-Net** | **DenseNet-201** | **512×512** | **0.5 µm** | **Lovász** | **Primary — closest to thesis best** |
| C | U-Net++ | EfficientNet-B3 | 512×512 | 0.5 µm | Dice | Dense skip connections |
| D | FPN | ResNet-50 | 512×512 | 0.5 µm | Lovász | Multi-scale features |

Config B is the primary baseline; target F1 ≥ 0.850.

### 4.3 YAML Class Mapping

`network_configuration.yaml` must be updated for 15 classes at 0.5 µm spacing:

```yaml
sampler:
  training:
    patch_shapes:
      0.5: [512, 512]
    label_map:         # XML integer → model class index
      1: 0   # WT-blastema
      2: 1   # WT-stroma
      3: 2   # WT-epithelium
      4: 3   # Necrosis
      5: 4   # Bleeding
      6: 5   # Regression (+ Fibrosis)
      7: 6   # Glomeruli
      8: 7   # Tubules
      9: 8   # Nephrogenic rests
      10: 9  # Fat
      11: 10 # Connective tissue
      12: 11 # Blood vessels
      13: 12 # Nerves
      14: 13 # Lymph nodes
      15: 14 # Urothelium
    label_dist:        # oversampling weights
      1: 5.0   # WT-blastema   (high clinical priority, sparse)
      2: 3.0   # WT-stroma
      3: 3.0   # WT-epithelium
      4: 3.0   # Necrosis
      5: 2.0   # Bleeding
      6: 2.0   # Regression
      7: 1.5   # Glomeruli
      8: 1.5   # Tubules
      9: 2.0   # Nephrogenic rests
      10: 1.0  # Fat
      11: 1.0  # Connective tissue
      12: 1.0  # Blood vessels
      13: 2.0  # Nerves
      14: 3.0  # Lymph nodes
      15: 2.0  # Urothelium
```

### 4.4 Data Preparation Steps

1. Mount Samba share inside Docker container (images + masks)
2. Rasterise 108 annotation XMLs → TIFF masks at 0.5 µm using `pathology-common` utilities
3. Merge Fibrose annotations into Regressie label during rasterisation
4. Generate patient-level 70/15/15 split → `data/wt_train.yaml`, `wt_val.yaml`, `wt_test.yaml`

### 4.5 Evaluation Target

```bash
python3 code/awesomedice.py \
  --input_mask_path "/output/predictions/*.tif" \
  --ground_truth_path "/data/masks/{image}.tif" \
  --classes "{'wt_blastema':1,'wt_stroma':2,'wt_epithelium':3,...}" \
  --spacing 0.5 --output_path /output/scores.yaml --all_cm
```

Targets: overall F1 ≥ 0.80; WT-blastema F1 ≥ 0.70.

---

## 5. Repository Structure

New repo `wt-tumor-segmentation`:

| Path | Content |
|------|---------|
| `pathology-segmentation-pipeline/` | Git submodule (main branch) |
| `configs/wt_network_configuration.yaml` | WT-specific training config |
| `data/wt_{train,val,test}.yaml` | Data split YAMLs |
| `scripts/rasterise_annotations.py` | XML → TIFF mask conversion |
| `scripts/train.sh` / `evaluate.sh` | Docker run wrappers |
| `results/` | Scores and logs (no large files) |

Model weights → HuggingFace Hub (`PierpaoloV93/pathology-segmentation-models`, family `wt-segmentation`).

---

## 6. Open Questions

| # | Question |
|---|----------|
| 1 | **Samba mount path** — exact SMB path needed for Docker run scripts |
| 2 | **Pixel spacing** — confirm 0.5 µm (thesis fidelity) vs 1.0 µm (pipeline default, faster iteration) |
| 3 | **Sparse vs sparse+dense** — start with sparse only (108 files) then add dense subfolders as second experiment? |
| 4 | **Downstream diagnosis classification** — in scope or segmentation only? |

---

## 7. Next Steps

- [ ] Create `wt-tumor-segmentation` GitHub repo, add pipeline submodule
- [ ] Write `scripts/rasterise_annotations.py` (XML → TIFF, merge Fibrosis into Regression)
- [ ] Generate patient-level data split YAMLs
- [ ] Write `configs/wt_network_configuration.yaml` for 15 classes @ 0.5 µm
- [ ] Write `scripts/train.sh` and `scripts/evaluate.sh`
- [ ] First run: Config B (U-Net + DenseNet-201), sparse annotations only
- [ ] Upload best weights to HuggingFace under `wt-segmentation` family

---

## 8. References

- van Alst, L. (2021). *Deep Learning for Segmentation and Classification of Wilms' Tumours*. MSc Thesis, Radboud University / Radboudumc.
- Pathology Segmentation Pipeline: `../projects/pathology-segmentation-pipeline` (main branch)
- Annotations: `annotations/` root, 108 usable XML files (ASAP format), 61 patients
