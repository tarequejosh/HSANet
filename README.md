# HSANet: Lightweight Hybrid Scale-Attention Network for Brain Tumor MRI Classification

[![Paper](https://img.shields.io/badge/Paper-HSANet_Journal_Paper.docx-blue)](./HSANet_Journal_Paper.docx)

**Authors:** Tareque Jamil Josh, Md. Minhazur Rahman Mim, Md. Aminur Rahman Joy, Md. Assaduzzaman, Sheak Rashed Haider Noori  
**Affiliation:** Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh  
**Correspondence:** Md. Assaduzzaman — assaduzzaman.cse@diu.edu.bd

## Overview

HSANet (Hybrid Scale-Attention Network) is a parameter-efficient deep learning model for four-class brain tumor classification on T1-weighted MRI. It combines:

- **EfficientNet-B3** multi-scale backbone
- **AMSM** — depthwise-separable Adaptive Multi-Scale Module with softmax-normalized four-branch fusion
- **EDL** — Evidential Deep Learning head for single-pass uncertainty estimation

Version 6 removes the redundant Dual Attention Module (DAM) used in version 5 after ablation showed stacked attention did not improve performance.

## Key results (BRISC 2025 test set)

| Metric | HSANet (v6) |
|--------|-------------|
| Accuracy | **99.2%** |
| Macro-F1 | **99.30%** |
| Parameters | **10.97M** |
| External (PMRAM-BD) | **97.41%** |

See [`results/`](./hsanet_results%20v6/results/) for full JSON metrics and [`figures/`](./hsanet_results%20v6/figures/) for publication plots.

## Repository structure

```
HSANet/
├── README.md
├── requirements.txt
├── hsanet-v6.ipynb          # Main experiment notebook (Kaggle-ready)
├── hsanet-v5.ipynb          # Earlier prototype (AMSM + DAM)
├── HSANet_Journal_Paper.docx
├── HSANet_Journal_Paper.md
├── generate_paper_docx.py
└── hsanet_results v6/
    ├── results/             # evaluation JSON
    └── figures/             # publication figures
```

## Datasets

| Dataset | Role | Link |
|---------|------|------|
| **BRISC 2025** | Primary (train/test) | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) · [DOI 10.34740/kaggle/ds/7632487](https://doi.org/10.34740/kaggle/ds/7632487) |
| **PMRAM-BD** | External validation | [Mendeley](https://data.mendeley.com/datasets/m7w55sw88b/1) |

BRISC was created by Fateh *et al.* (IUST, Shahrood University of Technology, NHS UK, University of Essex). This project uses the public Kaggle release and is not affiliated with the BRISC authors.

## Quick start

```bash
pip install -r requirements.txt
```

Open `hsanet-v6.ipynb` in Jupyter or upload to [Kaggle](https://www.kaggle.com/) with BRISC and PMRAM-BD datasets attached. Update `DATASET_PATHS` in the notebook if running locally.

## Citation

If you use this code or report, please cite:

```bibtex
@article{hsanet2026,
  title   = {HSANet: A Lightweight Hybrid Scale-Attention Network with Evidential Uncertainty for Brain Tumor MRI Classification},
  author  = {Josh, Tareque Jamil and Mim, Md. Minhazur Rahman and Joy, Md. Aminur Rahman and Assaduzzaman, Md. and Noori, Sheak Rashed Haider},
  affiliation = {Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh},
  year    = {2026},
  note    = {Code: https://github.com/tarequejosh/HSANet}
}
```

Also cite the BRISC dataset: Fateh *et al.*, *Scientific Data*, https://doi.org/10.1038/s41597-026-06753-y

## License

Code: MIT (see `LICENSE`). Dataset licenses follow BRISC (CC BY 4.0) and PMRAM-BD (CC BY 4.0) terms.

## Contact

Md. Assaduzzaman — assaduzzaman.cse@diu.edu.bd
