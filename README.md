# HSANet

Brain tumor MRI classification (4 classes) using EfficientNet-B3, depthwise-separable multi-scale attention (AMSM), and an evidential classifier.

**Team:** Tareque Jamil Josh, Md. Minhazur Rahman Mim, Md. Aminur Rahman Joy, Md. Assaduzzaman, Sheak Rashed Haider Noori  
**Dept. of CSE, Daffodil International University, Dhaka**  
Contact: assaduzzaman.cse@diu.edu.bd

## Results (BRISC 2025 test)

| Metric | HSANet v6 |
|--------|-----------|
| Accuracy | 99.2% |
| Macro-F1 | 99.30% |
| Params | 10.97M |
| PMRAM-BD (external) | 97.41% |

Metrics: `hsanet_results v6/results/` · Figures: `hsanet_results v6/figures/`

## Setup

```bash
pip install -r requirements.txt
```

Run `hsanet-v6.ipynb` (main). `hsanet-v5.ipynb` is an older build with an extra DAM attention block (removed in v6).

**Datasets**

- [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025) — train/test  
- [PMRAM-BD](https://data.mendeley.com/datasets/m7w55sw88b/1) — external test  

Set `DATASET_PATHS` in the notebook for local paths, or attach datasets on Kaggle.

## Layout

```
hsanet-v6.ipynb
hsanet-v5.ipynb
requirements.txt
hsanet_results v6/
  results/*.json
  figures/*.png
  models/README.md
```

Trained `.pth` checkpoints are not in the repo (size). See `hsanet_results v6/models/README.md`.

## Citation

BRISC dataset: Fateh et al., *Scientific Data*, https://doi.org/10.1038/s41597-026-06753-y

## License

MIT — see `LICENSE`.
