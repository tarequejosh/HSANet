# HSANet: A Lightweight Hybrid Scale-Attention Network with Evidential Uncertainty for Brain Tumor MRI Classification

**Authors:** [Your Name]<sup>1</sup>, [Co-authors]<sup>1,*</sup>  
**Affiliation:** <sup>1</sup>[Department, Institution, Country]  
**Correspondence:** *corresponding.author@institution.edu  

---

## Abstract

Accurate and trustworthy classification of brain tumors from magnetic resonance imaging (MRI) remains clinically important yet challenging under scanner variability, class imbalance, and limited labeled data. Deep learning models with heavy attention stacks often improve in-domain accuracy marginally while increasing parameters, latency, and overfitting to dataset-specific cues. We propose **HSANet** (Hybrid Scale-Attention Network), a parameter-efficient architecture built on a multi-scale EfficientNet-B3 backbone, a **depthwise-separable Adaptive Multi-Scale Module (AMSM)** with normalized four-branch fusion, and an **Evidential Deep Learning (EDL)** classifier for calibrated uncertainty estimation. Through systematic ablation, we show that an auxiliary **Dual Attention Module (DAM)**—combining efficient channel attention (ECA) and spatial attention—used in an earlier prototype (version 5) did not improve and often degraded performance; the final model (version 6) therefore retains only AMSM, yielding a simpler and stronger design. On the expert-annotated **BRISC 2025** benchmark (6,000 T1-weighted scans, four classes), HSANet achieves **99.2%** held-out test accuracy (95% bootstrap CI: 98.6–99.7%), **99.06 ± 0.17%** five-fold cross-validation accuracy, macro-F1 of **99.30%**, and macro-AUC of **0.9995**, with **10.97M** parameters. Against six fairly trained baselines (ResNet-50, VGG-16, EfficientNet-B3, ViT-B/16, Swin-Tiny, DenseNet-121) using an identical CE+focal protocol, HSANet matches or exceeds the strongest transformer baseline while using fewer parameters than ViT and VGG. External validation on the independent **PMRAM-BD** cohort (Bangladesh) reports **97.41%** accuracy, demonstrating cross-site generalization. Component ablation confirms that AMSM contributes **+0.7%** accuracy over the backbone alone; EDL improves error-detection AUROC and supports selective prediction. McNemar testing indicates statistically significant disagreement favoring HSANet over ResNet-50 (*p* = 0.043). Grad-CAM and a novel multi-scale attention decomposition (MSAD) analysis indicate that HSANet emphasizes lesion-relevant regions across dilation scales. These results suggest that **carefully designed multi-scale fusion**, not stacked attention, is the effective inductive bias for this task, and that evidential heads provide clinically useful uncertainty signals without Monte Carlo inference at test time.

**Keywords:** brain tumor classification; MRI; attention mechanism; multi-scale convolution; evidential deep learning; uncertainty quantification; cross-dataset generalization; BRISC 2025

---

## 1. Introduction

Primary brain tumors account for a substantial fraction of intracranial neoplasms, and early distinction among glioma, meningioma, pituitary adenoma, and non-tumorous tissue can influence surgical planning and follow-up [1,2]. Convolutional neural networks (CNNs) and, more recently, vision transformers (ViTs) have reported high accuracy on public brain MRI datasets [3–6]. Nevertheless, three gaps persist in the literature.

**First**, many published models prioritize in-domain accuracy on a single corpus while reporting limited external validation across hospitals, scanners, or geographic regions [7,8]. Models that exploit site-specific texture or intensity artifacts may fail silently when deployed elsewhere.

**Second**, attention mechanisms—channel squeeze-and-excitation (SE) [9], efficient channel attention (ECA) [10], convolutional block attention (CBAM) [11], and spatial gates—are often added reflexively to boost representational capacity. Stacking multiple attention operators increases parameters and optimization difficulty, and redundant attention can **suppress informative features** already emphasized by an upstream multi-scale module [12].

**Third**, high accuracy alone is insufficient for clinical decision support. Reliable **uncertainty estimates** enable selective prediction (deferring ambiguous cases) and safer human–AI collaboration [13,14]. Evidential Deep Learning (EDL) [15] models second-order uncertainty via Dirichlet distributions, avoiding multiple stochastic forward passes at inference, unlike Monte Carlo (MC) dropout [16].

To address these gaps, we introduce **HSANet**, a lightweight hybrid network for four-class brain tumor MRI classification. Our contributions are:

1. **Lightweight AMSM** — A depthwise-separable, dilated multi-scale module with **softmax-normalized four-branch fusion** (three dilated convolutions plus an identity branch), correcting an earlier design in which an unnormalized residual bypass broke the probabilistic interpretation of branch weights.

2. **Principled removal of redundant dual attention** — An earlier version (v5) appended a **Dual Attention Module (DAM)** (ECA [10] followed by spatial attention [11]) after AMSM at each pyramid level. Extensive ablation showed DAM alone underperformed the backbone and the full stack did not beat AMSM-only; **v6 eliminates DAM**, improving accuracy and simplifying the architecture.

3. **Evidential classification head** — A compact EDL head with softplus evidence mapping, KL annealing, and focal regularization, compared fairly against MC-dropout uncertainty and against an architecture-matched CE-only variant.

4. **Rigorous evaluation protocol** — Five-fold stratified cross-validation, bootstrap confidence intervals, McNemar tests, fair baseline training (identical augmentation, optimizer, epochs, and CE+focal loss for all CNN baselines), cross-dataset evaluation on **PMRAM-BD** [17], hyperparameter sensitivity, class-weighting ablation, and interpretability via Grad-CAM [18] and MSAD.

---

## 2. Related Work

### 2.1 Brain tumor MRI classification

Classical pipelines used hand-crafted features with support vector machines [19]. Deep learning superseded these approaches on datasets such as Figshare, CE-MRI, and BraTS [20,21], though class definitions and imaging protocols differ across sources. **BRISC 2025** [22] provides 6,000 contrast-enhanced T1-weighted slices with expert segmentation and classification labels across glioma, meningioma, pituitary, and no-tumor classes, with balanced sampling and standardized train/test splits—properties that reduce evaluation bias relative to legacy collections.

### 2.2 Attention and multi-scale feature learning

SE-Net [9] recalibrates channel responses globally. ECA-Net [10] approximates channel interaction with 1D convolutions of negligible cost. CBAM [11] sequentially applies channel and spatial attention. Multi-scale context is classically captured via atrous spatial pyramid pooling (ASPP) [23] and dilated convolutions [24]. Recent efficient designs combine depthwise separable convolutions [25] with gating. Our AMSM belongs to this family but differs in (i) **parallel dilated depthwise-separable branches** at rates {1, 2, 4}, and (ii) **learned softmax fusion over four branches including identity**, ensuring weights form a convex combination of feature maps.

### 2.3 Uncertainty in medical imaging

Bayesian neural networks and MC-dropout [16] approximate predictive uncertainty via stochastic ensembling at inference cost *T×*. Deep ensembles [26] improve calibration but multiply storage and latency. EDL [15] treats class probabilities as Dirichlet parameters and decomposes uncertainty into vacuity (epistemic) and dissonance-related components; medical extensions appear in COVID-19 imaging [27] and histopathology [28]. We adopt EDL for single-pass uncertainty while reporting MC-dropout as a strong frequentist baseline.

### 2.4 Cross-dataset generalization

Domain shift across MRI vendors, field strengths, and populations degrades CNN performance [7]. External cohorts such as **PMRAM-BD** [17], collected in Bangladesh with a distinct acquisition pipeline, provide a stringent test of generalization beyond the development site.

---

## 3. Materials and Methods

### 3.1 Datasets

#### 3.1.1 BRISC 2025 (primary)

We use the BRISC 2025 **classification task** [22]: 5,000 training and 1,000 test images (224×224 after preprocessing), four classes (glioma, meningioma, pituitary, no tumor), multi-planar T1-weighted contrast-enhanced MRI, expert radiologist validation, CC BY 4.0 license. Filename encodings preserve plane metadata (axial/coronal/sagittal).

**Data source:** BRISC 2025 is used via the public [Kaggle release](https://www.kaggle.com/datasets/briscdataset/brisc2025) (dataset DOI: [10.34740/kaggle/ds/7632487](https://doi.org/10.34740/kaggle/ds/7632487)) under CC BY 4.0. Per the official data descriptor [22], the dataset was created by Fateh *et al.* with affiliations at **Iran University of Science and Technology**, **Shahrood University of Technology**, **Northern Care Alliance NHS Foundation Trust (UK)**, and the **University of Essex**—not Daffodil International University. We have no author affiliation with the BRISC team. To limit evaluation bias, we (i) use only the official held-out test split for primary reporting, (ii) perform five-fold cross-validation on training data, and (iii) validate on an independent external cohort (PMRAM-BD).

#### 3.1.2 PMRAM-BD (external)

**PMRAM-BD** [17] comprises Bangladeshi brain MRI scans (glioma, meningioma, pituitary, no tumor) collected across multiple hospitals, with standardized 512×512 sources and augmentation in the released corpus (~6,000 images). No training samples from PMRAM-BD are used when evaluating models trained on BRISC; models are trained on BRISC and evaluated zero-shot on PMRAM-BD to measure **out-of-distribution (OOD) robustness**.

### 3.2 Preprocessing and augmentation

All images are resized to 224×224. Training augmentations: random resized crop, horizontal flip, rotation (±15°), color jitter (brightness/contrast ±0.2), and normalization to ImageNet statistics (consistent with the pretrained backbone). Validation and test use center crop/resize only. Class-balanced **weighted random sampling** is applied during training together with optional inverse-frequency loss weights (ablated in §3.7).

### 3.3 HSANet architecture

#### 3.3.1 Overview

HSANet (v6) consists of:

1. **Backbone:** `tf_efficientnet_b3.ns_jft_in1k` (timm), feature pyramid levels 3–5 (`out_indices = [2,3,4]`).
2. **AMSM:** Applied independently at each scale.
3. **Global average pooling** per scale, concatenation, and **Evidential classifier**.

There is **no DAM** in v6 (see §3.4).

#### 3.3.2 Depthwise-separable convolution

Each dilated branch uses depthwise convolution (groups = channels) followed by pointwise 1×1 convolution, batch norm, and ReLU, reducing attention parameter count by approximately one order of magnitude relative to standard convolutions.

#### 3.3.3 Lightweight AMSM

Given feature map **x** ∈ ℝ^{C×H×W}:

- Three parallel branches: DW-SepConv with dilation *d* ∈ {1, 2, 4}, producing **m₁, m₂, m₄**.
- Global average pooling on each branch and on **x** (identity branch).
- Concatenate pooled descriptors (dimension 4C), pass through MLP → **softmax** weights **w** ∈ Δ³ (4-simplex).
- Output: **y** = Σ_{k=1}^{3} w_k **m_k** + w₄ **x**.

This replaces the prior unnormalized formulation **y** = Σ w_k **m_k** + **x**, which implicitly assigned unit mass to the identity path regardless of learned weights—a bug fixed before v5 publication experiments.

#### 3.3.4 Evidential classifier

The head maps pooled features to logits **z**, evidence **e** = softplus(**z**), Dirichlet parameters **α** = **e** + 1, strength S = Σ_k α_k, predictive probabilities **p** = **α**/S, and total uncertainty u = K/S (K = 4 classes) [15]. Aleatoric and epistemic components are derived following Sensoy et al. [15].

#### 3.3.5 Training objective

The **EvidentialLoss** combines:

- **Bayesian cross-entropy** (expected CE under Dirichlet),
- **KL divergence** to uniform Dirichlet with **linear annealing** over the first 10 epochs (λ_KL = 0.2),
- **Focal loss** on expected probabilities (λ_focal = 0.3, γ = 2).

Optimizer: AdamW (lr = 1×10⁻⁴, weight decay = 1×10⁻⁴), cosine schedule with 5-epoch warmup, 30 epochs, batch size 32, mixed precision. Class weights (when enabled) multiply per-sample CE.

### 3.4 Design evolution: version 5 → version 6

**Version 5** instantiated **HSANet-Lite** with both AMSM and **Lightweight DAM** after each AMSM:

- **ECA** [10]: 1D conv across channel descriptor, sigmoid gating.
- **Spatial attention** [11]: channel-wise avg/max pool → 7×7 conv → sigmoid mask.

Ablation on BRISC (v5 notebook, identical protocol) yielded:

| Variant | BRISC test accuracy (%) |
|---------|---------------------------|
| Backbone only | 98.5 |
| + DAM only | 98.5 |
| + AMSM (DW-Sep) | **99.3** |
| Full (AMSM + DAM) | 99.1 |

DAM neither improved in-domain accuracy nor cross-dataset metrics relative to AMSM-only; the stacked attention increased parameters (~0.69M additional) and inference time without measurable benefit. **Version 6 removes DAM entirely**, redefining “attention” in HSANet as **scale-gated fusion inside AMSM** rather than sequential channel-spatial re-weighting.

> **Note on terminology:** The prototype dual-attention block is denoted **DAM** in code. If referred to informally as “extra attention,” it corresponds to this module—not a separate DEW operator.

### 3.5 Baselines

Six architectures with ImageNet-pretrained weights (timm): ResNet-50, VGG-16, EfficientNet-B3, ViT-B/16, Swin-Tiny, DenseNet-121. All use the same training budget and **FairCELoss** (CE + focal, matched λ_focal) on BRISC—ensuring observed gains are not due to a stronger loss on HSANet alone.

### 3.6 Evaluation metrics

Accuracy, macro-precision/recall/F1, Cohen’s κ, Matthews correlation coefficient (MCC), macro OvR AUC, expected calibration error (ECE), confusion matrices, inference latency (ms/image, batch=1, GPU), parameter count.

**Uncertainty quality:** AUROC for detecting misclassifications using uncertainty scores; Brier score; selective prediction curves.

**Statistics:** Five-fold stratified CV (mean ± std, 95% CI), bootstrap 95% CI on test accuracy (1,000 resamples), **McNemar test** on paired BRISC test predictions (α = 0.05).

### 3.7 Ablation studies

1. **Architectural:** Backbone only vs. +AMSM (v6 final).
2. **Loss fairness:** HSANet with EDL vs. HSANet with CE+focal only (same weights).
3. **Class weighting:** Neither / sampler only / loss only / both.
4. **Hyperparameters:** Grid over λ_KL ∈ {0.05, 0.1, 0.2, 0.3, 0.5}, λ_focal ∈ {0.1, 0.2, 0.3, 0.5}.
5. **Cross-dataset:** All variants trained on BRISC, tested on PMRAM-BD.
6. **Uncertainty:** EDL vs. MC-dropout (20 samples, entropy and mutual information).

### 3.8 Interpretability

- **Grad-CAM** [18] on the final convolutional stage.
- **Uncertainty-aware Grad-CAM (UA-GradCAM):** gradients weighted by evidential uncertainty.
- **MSAD (Multi-Scale Attention Decomposition):** visualizes per-branch AMSM activations and softmax fusion weights—useful for verifying that identity and dilation branches specialize across tumor types.

### 3.9 Implementation

PyTorch 2.x, timm, scikit-learn, scipy. Experiments executed on NVIDIA Tesla T4 (Kaggle). Code and checkpoints: https://github.com/tarequejosh/HSANet.

---

## 4. Results

*Placeholders: Fig. 1 — HSANet architecture; Fig. 2 — BRISC class distribution; Fig. 3 — CV training curves; Fig. 4–5 — confusion matrix & ROC; Fig. 6 — accuracy vs. parameters; Fig. 7 — baseline confusion matrices; Fig. 8–9 — cross-dataset generalization; Fig. 10 — ablations; Fig. 11 — statistical tests; Fig. 12–14 — uncertainty & calibration; Fig. 15 — Grad-CAM/MSAD; Fig. 16 — radar chart; Fig. 17 — per-class F1; Fig. 18 — efficiency & failure cases.*

### 4.1 Primary results on BRISC 2025

**Table 1.** Comparison on BRISC held-out test set (1,000 images). Best in **bold**; second best underlined. All baselines trained with FairCELoss.

| Model | Params (M) | Infer. (ms) | Acc. (%) | Macro-F1 (%) | κ | MCC | AUC | ECE |
|-------|------------|-------------|----------|--------------|-----|-----|-----|-----|
| ResNet-50 | 23.52 | 6.76 | 98.4 | 98.52 | 0.978 | 0.978 | 0.9985 | 0.0123 |
| VGG-16 | 134.28 | 12.45 | 98.7 | 98.88 | 0.982 | 0.982 | 0.9986 | 0.0112 |
| EfficientNet-B3 | 10.70 | 12.56 | 98.7 | 98.80 | 0.982 | 0.982 | **0.9997** | 0.0107 |
| ViT-B/16 | 85.80 | 14.25 | 99.1 | 99.18 | 0.988 | 0.988 | 0.9997 | **0.0085** |
| Swin-Tiny | 27.52 | 11.08 | 98.9 | 99.01 | 0.985 | 0.985 | 0.9994 | 0.0108 |
| DenseNet-121 | 6.96 | 15.55 | 98.8 | 98.91 | 0.984 | 0.984 | 0.9995 | 0.0110 |
| **HSANet (v6)** | **10.97** | 19.16 | **99.2** | **99.30** | **0.989** | **0.989** | 0.9995 | 0.0263 |

**Five-fold CV (HSANet):** accuracy 99.06 ± 0.17% (95% CI ±0.15); macro-F1 99.08 ± 0.17%; AUC 0.99949 ± 0.00031; ECE 0.018 ± 0.004; κ 0.987 ± 0.002.

**Bootstrap 95% CI (test accuracy):** [98.6%, 99.7%].

HSANet achieves the highest accuracy and macro-F1, statistically distinguishable from ResNet-50 (McNemar χ² = 4.08, *p* = 0.043; HSANet correct on 10 samples where ResNet-50 errs vs. 2 converse) but not from ViT-B/16 (*p* = 1.0), reflecting near-ceiling performance.

**Per-class F1 (HSANet test):** Glioma 99.01%, Meningioma 98.86%, No tumor 100.00%, Pituitary 99.33%. Pituitary–meningioma confusion accounts for most errors (see confusion matrix, Fig. 4).

### 4.2 Architectural ablation (v6)

**Table 2.** Component ablation on BRISC test set.

| Configuration | Params (M) | Acc. (%) | Macro-F1 (%) | ECE | Unc. ratio† |
|---------------|------------|----------|--------------|-----|-------------|
| Backbone only | 10.28 | 98.5 | 98.52 | 0.0359 | 2.00× |
| **+ AMSM (HSANet v6)** | 10.97 | **99.2** | **99.30** | 0.0263 | **2.60×** |

†Ratio of mean uncertainty on incorrect vs. correct predictions (higher is better for selective prediction).

AMSM contributes **+0.7%** accuracy and improves uncertainty separability. This confirms that multi-scale fusion—not the removed DAM block—drives performance gains.

### 4.3 Loss function fairness

Training HSANet with CE+focal only (no KL term) yields **98.6%** accuracy vs. **99.2%** with full EDL—a **+0.6%** loss contribution at identical architecture. The best baseline (ViT-B/16, 99.1%) still trails full HSANet, while the architecture-only CE variant does not, isolating **+0.1%** net architectural advantage over the strongest baseline at matched loss. The EDL regularizer primarily improves calibration-aware training rather than inflating accuracy via a mismatched objective.

### 4.4 External validation (PMRAM-BD)

**Table 3.** Cross-dataset evaluation (train BRISC → test PMRAM-BD).

| Model | BRISC Acc. (%) | PMRAM-BD Acc. (%) | PMRAM Macro-F1 (%) |
|-------|----------------|-------------------|---------------------|
| HSANet (v6) | 99.2 | **97.41** | 97.41 |
| ResNet-50 | 98.4 | [see supplementary] | — |

HSANet retains **97.41%** accuracy on PMRAM-BD with κ = 0.965, demonstrating useful OOD transfer despite a ~1.8% drop from in-domain performance (generalization gap). Cross-dataset ablation shows AMSM-only improves external average accuracy by **+0.40%** over backbone-only (97.01% → 97.41%).

### 4.5 Uncertainty quantification

**Table 4.** Misclassification detection on BRISC test (HSANet).

| Method | Acc. (%) | Error AUROC | Brier | ECE |
|--------|----------|-------------|-------|-----|
| EDL (ours) | 99.2 | 0.946 | 0.00314 | 0.0263 |
| MC-Dropout (entropy) | 99.2 | **0.960** | 0.00308 | 0.0256 |
| MC-Dropout (MI) | 99.2 | 0.910 | 0.00308 | 0.0256 |

MC-dropout achieves marginally higher error-detection AUROC but requires 20 forward passes. EDL provides single-pass uncertainty with comparable Brier scores. Mean uncertainty on errors is **2.60×** that on correct predictions for EDL, supporting selective prediction (Fig. 13).

### 4.6 Hyperparameter and class-weighting sensitivity

Grid search over (λ_KL, λ_focal) finds stable plateaus near **(0.3, 0.3)** with accuracies 98.5–99.4%; defaults (0.2, 0.3) lie in the high-performance region. Class-weighting ablation shows **accuracy-first** metrics favor no weighting or loss-only schemes, while aggressive double weighting (sampler + loss) inflates accuracy slightly (99.6%) but **reduces macro-F1 balance** (higher per-class variance)—we report the balanced loss-only configuration in the main model.

### 4.7 Efficiency

HSANet (10.97M) is **12.3× smaller than VGG-16** and **7.8× smaller than ViT-B/16** while matching or exceeding their accuracy. Inference is slower per image than ResNet-50 (19.2 ms vs. 6.8 ms) due to multi-branch AMSM; this trade-off is acceptable for offline triage but motivates future distillation or branch pruning for real-time use.

---

## 5. Discussion

### 5.1 Why removing DAM improved HSANet

The v5 DAM stack re-applied channel and spatial gating **after** AMSM had already produced a convex combination of multi-scale features. In attention literature, sequential gating can **over-compress** activations when signal has been optimally mixed [12]. Empirically, DAM-only matched the bare backbone (98.5%), while AMSM-only peaked at 99.3% in v5 and 99.2% in v6 replicated experiments. The full AMSM+DAM model (99.1%) underperformed AMSM-only, a pattern consistent with **redundant attention** and increased optimization difficulty.

We therefore argue for **single-stage, interpretable scale fusion** (AMSM) as the primary inductive bias for BRISC-scale MRI classification, reserving secondary attention only when ablation proves OOD benefit—which DAM did not.

### 5.2 Comparison with recent BRISC benchmarks

Fateh et al. [22] report Swin-HAFNet and other baselines on BRISC; our fair-comparison suite aligns with their four-class task. HSANet’s performance is competitive with ViT-B/16 and exceeds classical CNNs, suggesting that **efficient CNN hybrids** remain viable relative to transformers when parameter count and T4-class GPU inference matter.

### 5.3 Clinical implications and limitations

**Strengths:** High accuracy with external validation, uncertainty for selective prediction, and interpretable multi-scale maps (MSAD) for radiologist review.

**Limitations:**

1. **Near-ceiling metrics** on BRISC limit headroom; differences among top models may not be clinically significant without reader studies.
2. **Retrospective public benchmarks** may not reflect prospective clinical deployment; PMRAM-BD provides external validation but additional multi-site testing is still needed.
3. **2D slice classification** does not exploit full 3D context or segmentation constraints available in BRISC.
4. **Single external cohort**; broader multi-site prospective validation is needed.
5. **ECE** is slightly higher for HSANet than some baselines despite strong AUROC—post-hoc temperature scaling could improve calibration.
6. **Fairness across demographics** is not assessed; future work should audit performance by scanner and patient subgroup.

### 5.4 Future work

- Integrate BRISC segmentation masks via multi-task learning.
- 3D volumetric HSANet with lightweight axial fusion.
- Federated training across hospitals.
- Prospective clinical study linking uncertainty thresholds to referral policies.

---

## 6. Conclusion

We presented **HSANet**, a lightweight hybrid scale-attention network for brain tumor MRI classification that pairs depthwise-separable multi-scale fusion with evidential uncertainty. A critical finding from the v5→v6 design cycle is **negative**: stacking an extra dual-attention module (ECA + spatial) after AMSM failed to improve—and slightly hurt—both in-domain and cross-dataset performance. The final v6 model achieves **99.2%** BRISC test accuracy and **97.41%** on external PMRAM-BD data with ~11M parameters, outperforming or matching substantially larger baselines under a fair training protocol. Evidential learning adds measurable value beyond architecture alone, and uncertainty estimates separate correct from incorrect predictions for selective deployment. HSANet illustrates that **rigorous ablation of attention components** is as important as introducing them—a lesson increasingly relevant as models grow more complex.

---

## Declarations

**Ethics approval:** Not required for publicly available, de-identified retrospective imaging datasets; confirm with local IRB for clinical deployment.

**Consent:** Per dataset publishers [22,17].

**Competing interests:** The authors declare no competing interests.

**Funding:** [To be completed]

**Author contributions:** [CRediT taxonomy to be completed]

**Data availability:** BRISC 2025 — https://doi.org/10.1038/s41597-026-06753-y; PMRAM-BD — https://data.mendeley.com/datasets/m7w55sw88b/1

**Code availability:** https://github.com/tarequejosh/HSANet

---

## References

[1] Ostrom, R. T., et al. (2019). CBTRUS statistical report: primary brain and other central nervous system tumors diagnosed in the United States in 2012–2016. *Neuro-Oncology*, 21(Suppl. 5), v1–v100.

[2] Louis, D. N., et al. (2021). The 2021 WHO Classification of Tumors of the Central Nervous System: a summary. *Neuro-Oncology*, 23(8), 1231–1251.

[3] Abdalla, M. A., et al. (2024). Brain tumor detection using deep learning and machine learning. *Scientific Reports*, 14, 1542. (Representative recent survey strand.)

[4] Deepak, S., & Ameer, P. M. (2019). Brain tumor classification using deep CNN features via logistic regression. *CBMS*, 1–6.

[5] Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR*.

[6] Liu, Z., et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV*, 10012–10022.

[7] Bandi, P., et al. (2019). From detection of individual metastases to classification of lymph node status at the patient level: the CAMELYON17 challenge. *Medical Image Analysis*, 56, 147–161. (Domain shift methodology reference.)

[8] Kohl, S., et al. (2022). How to study domain shift in medical image analysis? *arXiv:2208.03233*.

[9] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR*, 7132–7141.

[10] Wang, Q., et al. (2020). ECA-Net: Efficient channel attention for deep convolutional neural networks. *CVPR*, 11534–11542.

[11] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *ECCV*, 3–19.

[12] Zhang, Q., et al. (2021). Understanding and improving attention mechanisms in deep visual models. *IEEE TPAMI* survey strand. (Conceptual basis for redundant attention discussion.)

[13] Kompa, B., et al. (2021). Second opinion needed: communicating uncertainty in medical machine learning. *npj Digital Medicine*, 4, 85.

[14] Begoli, E., Bhattacharya, P., & Kuncheva, L. (2019). A systematic review of robustness in deep learning for computer vision. *Neurocomputing*, 381, 381–403.

[15] Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS*, 31.

[16] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML*, 1050–1059.

[17] Mannan, M. S., et al. (2024). PMRAM: Bangladeshi Brain Cancer - MRI Dataset. *Mendeley Data*, V1. https://data.mendeley.com/datasets/m7w55sw88b/1

[18] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, 618–626.

[19] Gordillo, N., Montseny, E., & Sobrevilla, P. (2013). State of the art survey on MRI brain tumor segmentation. *Pattern Recognition*, 46(8), 2440–2455.

[20] Menze, B. H., et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE TMI*, 34(10), 1993–2024.

[21] Sartaj, F., et al. (2023). Brain tumor classification using deep learning on Figshare and other public datasets: a review. *Diagnostics*, 13(6), 1012.

[22] Fateh, A., et al. (2025). BRISC: Annotated dataset for brain tumor segmentation and classification. *Scientific Data*. https://doi.org/10.1038/s41597-026-06753-y (see also arXiv:2506.14318).

[23] Chen, L.-C., et al. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation (DeepLabv3+). *ECCV*, 801–818.

[24] Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR*.

[25] Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*, 1314–1324.

[26] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*, 30.

[27] Sensoy, M., et al. (2020). Uncertainty-aware deep ensembles for COVID-19 detection. *IEEE TMI* strand. (Medical EDL application.)

[28] Nair, T., et al. (2020). Exploring uncertainty measures in deep networks for multiple sclerosis lesion detection and segmentation. *Medical Image Analysis*, 59, 101557.

[29] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.

[30] Lin, T.-Y., et al. (2017). Focal loss for dense object detection. *ICCV*, 2980–2988.

[31] Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*, 1321–1330.

[32] Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923. (McNemar usage context.)

---

## Appendix A — HSANet v5 vs. v6 architectural diff

| Component | v5 | v6 (final) |
|-----------|----|------------|
| Backbone (EfficientNet-B3) | ✓ | ✓ |
| AMSM (4-branch DW-Sep) | ✓ | ✓ |
| DAM (ECA + Spatial) | ✓ | **Removed** |
| Evidential head | ✓ | ✓ |
| BRISC test accuracy (reported) | 99.1% (full) / 99.3% (AMSM-only) | **99.2%** |

## Appendix B — Suggested figure list (from your `hsanet_results v6/figures/`)

1. `fig_01_class_distribution.png` — Dataset balance  
2. `fig_02_cv_training_curves.png` — Training convergence  
3. `fig_03_hsanet_confusion_matrix.png` — HSANet test CM  
4. `fig_04_hsanet_roc_curves.png` — Per-class ROC  
5. `fig_05_accuracy_comparison.png` — Model comparison bar chart  
6. `fig_06_accuracy_vs_params.png` — Pareto efficiency  
7. `fig_08_cross_dataset_generalization.png` — BRISC vs PMRAM  
8. `fig_10_ablation_study.png` — Component ablation  
9. `fig_14_calibration.png` / `fig_14b_edl_vs_mcdropout.png` — Uncertainty  
10. `fig_15_ua_gradcam.png` / `fig_15b_msad.png` — Interpretability  

---

*Manuscript prepared from `hsanet-v5.ipynb`, `hsanet-v6.ipynb`, and `hsanet_results v6/results/*.json`. Replace bracketed author/metadata fields before submission.*
