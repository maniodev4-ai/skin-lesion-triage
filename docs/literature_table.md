# Literature review, research gap and measurable targets (Part A)

**Verification status.** Every paper below was opened on its publisher, PubMed, PMC or arXiv page in this session, and the authors, venue and the facts quoted were read there. Two kinds of entries are marked:
- **(preprint)**: arXiv only. Fine to cite, but cite it as a preprint.
- **(via ref. list)**: bibliographic details were read from another paper's reference list, but the paper itself was not opened. Read its abstract before the viva.

**AI-assistance disclosure for the report:** "An AI assistant (Claude, Anthropic) was used to search for and summarise literature, draft the literature table, and generate code. The authors reviewed the cited sources and are able to explain all code."

## 1. Papers (compared by method, data, protocol, result)

| # | Reference | Type | Data and protocol | Result or point used here |
|---|---|---|---|---|
| 1 | P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Sci. Data*, vol. 5, 180161, 2018. | Journal | 10,015 images, 7 classes, from Vienna and Australia. More than 50% confirmed by pathology; the rest by follow-up, expert consensus or confocal microscopy. | Dataset and label provenance (B.ii). |
| 2 | N. Gessert, M. Nielsen, M. Shaikh, R. Werner, A. Schlaefer, "Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data," *MethodsX*, vol. 7, 100864, 2020. | Journal | ISIC 2019, first place. Five folds with all images of a lesion in the same fold. Metadata is partly incomplete. CNN trained first, then frozen, then a dense metadata network and the classifier are trained. Weighted loss for imbalance. | Strong conventional design (Design B template). Shows lesion-grouped folds are standard in strong work. |
| 3 | A. G. C. Pacheco, R. A. Krohling, "The impact of patient clinical information on automated skin cancer detection," *Comput. Biol. Med.*, vol. 116, 103545, 2020. | Journal | New dataset of smartphone clinical images (not dermoscopy) with patient information. Compares image-only and image-plus-clinical models using a fusion scheme that controls each source's contribution. | Patient information helps detection. Image type differs from ours, so state that in the comparison. |
| 4 | J. Höhn et al., "Integrating patient data into skin cancer classification using convolutional neural networks: systematic review," *J. Med. Internet Res.*, July 2021 (PMID 34255646). | Journal | Systematic review, 11 studies. | All 11 reported improvement from patient data. Age, sex and lesion location were most used, mostly one-hot encoded. How patient data help in multi-class tasks is "still unclear". Supports our metadata design (C.ii). |
| 5 | A. Bissoto, M. Fornaciali, E. Valle, S. Avila, "(De)constructing bias on skin lesion datasets," *CVPR Workshops*, 2019. | Workshop | Experiments on ISIC and Atlas data with the lesion information removed. | Models can classify without clinically meaningful content. Motivates the shortcut checks (E.iv). |
| 6 | A. Bissoto, E. Valle, S. Avila, "Debiasing skin lesion datasets and models? Not so fast," *ISIC Skin Image Analysis Workshop, CVPR*, 2020. | Workshop | Hand annotation of 7 visual artifacts on ISIC 2018 and the Atlas. Tests a debiasing method. | Debiasing methods are "not ready"; artifacts can be exploited. Supports honest explanation claims. |
| 7 | J. Carse, T. Suveges, S. Hogg, E. Trucco, C. Proby, C. Fleming, S. McKenna, "Robust selective classification of skin lesions with asymmetric costs," *UNSURE 2021*, LNCS, doi:10.1007/978-3-030-87735-4_11. | Workshop | ISIC 2019, benign vs malignant. EC-SelectiveNet and others, scored by empirical coverage and selective cost under asymmetric costs. Temperature chosen on validation. | Closest prior work for R2 and R3. Binary task and fixed cost ratios, unlike our 7-class, unknown-r setting. |
| 8 | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern neural networks," *Proc. ICML*, PMLR vol. 70, pp. 1321-1330, 2017. | Conference | General (not skin-specific). | Source for temperature scaling, ECE and reliability diagrams. |
| 9 | K. Abhishek, A. Jain, G. Hamarneh, "Investigating the quality of DermaMNIST and Fitzpatrick17k dermatological image datasets," *Sci. Data*, 2025, doi:10.1038/s41597-025-04382-5. | Journal | Analyses DermaMNIST, its source HAM10000, and Fitzpatrick17k for duplicates, leakage across partitions, and mislabelling. | Evidence for the leakage and data-quality problems in B.ii and the failure-chain experiment in F.i. |
| 10 | A. Pınar et al., "A reproducible and explainable deep learning framework for dermoscopic image analysis in skin cancer," *J. Clin. Med.*, vol. 15, no. 19, 7422, 2026, doi:10.3390/jcm15197422. | Journal | HAM10000, lesion-level StratifiedGroupKFold (5 folds), temperature scaling on an inner validation partition, never the test fold. External test on BCN20000. | Closest protocol match to ours, and published numbers for E.v (section 4). Finds six architectures statistically indistinguishable and a 25-point balanced-accuracy drop externally. Also finds Grad-CAM-type maps score below a fixed centred disc on mask-IoU, so explanations need care. |
| 11 | P. K. Mondal, S. K. Das, S. K. Debnath, I. Mandal, M. S. Al Mamun, "Retrieval familiarity for selective referral in dermoscopic classification: an external evaluation under distribution shift," *IJACSA*, vol. 17, no. 10, pp. 1239-1251, 2026. | Journal | Lesion-disjoint HAM10000 partitions. Threshold chosen on validation for 50% coverage, then frozen. External tests on ISIC 2019 (14,885 images) and Derm7pt (963). | A ranking score that looks good by AURC did not keep its operating point: the 50% coverage threshold gave 21.95% on ISIC 2019 and 39.36% on Derm7pt, and most other scores also undershot. Intervals resample images, not lesions, and the authors say this may be too narrow. Direct support for the gap. |
| 12 | A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei, "Learn then Test: calibrating predictive algorithms to achieve risk control," arXiv:2110.01052. | Preprint | Chooses a model parameter (such as a threshold) so a chosen risk is controlled with finite-sample guarantees, framed as multiple hypothesis testing. | Method basis for the novel contribution. |
| 13 | S. J. McKenna, J. Carse, "Calibrating where it matters: constrained temperature scaling," arXiv:2406.11456. | Preprint | ISIC 2019, 60/20/20 split, EfficientNet-B7 and ResNet101. Fits temperature only on the region of predictions that can affect decisions; binary and multi-class. Reports ECE separately inside and outside the decision region. | Calibration should target the decision region, which for us is the referral threshold. |
| 14 | J. Traub et al., "Overcoming common flaws in the evaluation of selective classification systems," *NeurIPS*, vol. 37, pp. 2323-2347, 2024. | Conference (via ref. list) | General selective classification. | Cited by Mondal et al. for the point that AURC alone is not enough. Read the abstract before citing. |
| 15 | J. Naushad, I. Voiculescu, "Super-TrustScore: reliable failure detection for automated skin lesion diagnosis," *IEEE ISBI*, 2024, doi:10.1109/ISBI56570.2024.10635815. | Conference (via ref. list) | Skin lesion failure detection with global and class-conditional neighbourhoods. | Skin-specific failure detection for the deferral discussion. Read the abstract before citing. |
| 16 | J. Fayyad, S. Alijani, H. Najjaran, "Empirical validation of conformal prediction for trustworthy skin lesions classification," arXiv:2312.07460. | Preprint | Compares conformal prediction, MC dropout and evidential deep learning on skin lesions, with HAM10000 sample analysis. | Conformal methods have been applied to skin lesions as prediction sets and uncertainty. Not as a guaranteed referral sensitivity. |

**Peer-reviewed count:** papers 1 to 11, 14 and 15 are journal, conference or workshop papers (13). Papers 12, 13 and 16 are preprints. The rubric asks for at least 10 peer-reviewed papers, so you have margin.

## 2. What the literature shows (for the critical comparison)

- **Protocol.** Strong work groups folds by lesion (2, 10, 11). Leakage across partitions in HAM10000 is documented (9). Scores from image-level and lesion-level splits are not directly comparable, so compare only with lesion-grouped results.
- **Metadata.** Fusing patient information helped in all 11 studies reviewed (4), and the winning ISIC 2019 system used a metadata branch (2). In our baselines, image plus metadata beat image only on validation, with overlapping intervals.
- **Imbalance.** Weighted losses are the common remedy (2). We use balanced class weights.
- **Calibration.** Temperature scaling on a validation partition is standard (8, 10). Calibration aimed at the decision region is proposed (13).
- **Deferral.** Selective classification with asymmetric costs exists for binary ISIC 2019 (7). Ranking quality is usually scored by AURC (11, 14). Operating-point transfer is a separate and weaker point (11).
- **Bias and shortcuts.** Artifacts can drive predictions (5, 6), and saliency maps can fail a trivial baseline (10). Explanations must be presented with those limits.

## 3. Research gap

Existing selective-classification work for skin lesions chooses the referral threshold on a held-out set and reports it (7), scores the ranking by AURC (11), or gives conformal prediction-set guarantees (16). The closest evidence of the problem is Mondal et al. (11): thresholds fixed on validation failed to preserve their intended workload, though that study measured coverage under dataset shift.

I did not find a lesion-grouped HAM10000 study that asks the question this triage task needs answered: **if the target is a minimum referral sensitivity for the high-concern group, does a validation-chosen threshold deliver it on unseen lesions, and does a statistically controlled threshold fix that at an acceptable referral cost?** This is a claim about the papers I searched, not about all literature. Phrase it in the report as "to our knowledge".

The concern is concrete in your data. The validation set has about 280 high-concern images (19.6% of 1,431). In your baselines, referral sensitivity had a 95% lesion-bootstrap interval of roughly ±0.07. A threshold tuned to just reach a target on a set this small will often miss it on new lesions.

## 4. Research questions

- **RQ1.** Over many lesion-grouped resplits, how often does a naive validation-chosen referral threshold fall below its sensitivity target on held-out lesions, and by how much?
- **RQ2.** Does a risk-controlled threshold (finite-sample bound on the miss rate, Learn then Test style, counting lesions rather than images) meet the target with the stated confidence, and what extra referral rate does it cost?
- **RQ3.** A threshold chosen to hit a sensitivity target is unaffected by any monotone recalibration of the score, because the ranking does not change. Calibration matters for the cost-based rule instead: does calibrating P(high-concern) change the referral rate, sensitivity and cost of the threshold 1/(1+r)?
- **RQ4.** Do the answers hold when patient metadata is missing?

## 5. Novel contribution (Part D)

**Risk-controlled referral threshold for lesion-grouped triage.** Compare the conventional design (threshold chosen on validation to reach the target) against the same classifier with a statistically controlled threshold. Test by repeated lesion-grouped resplits of the train-plus-validation pool, so the locked test set is used once at the end.

Ablation ladder: (a) naive validation threshold, (b) plus a binomial lower confidence bound on sensitivity, (c) plus lesion-level counting, (d) analysed separately: calibration of P(high-concern) for the cost-based threshold 1/(1+r), since calibration cannot change a sensitivity-targeted threshold. Report target-miss rate, referral rate and intervals across resplits. If control costs too much referral, or the naive threshold turns out fine, say so (rubric D.iii).

## 6. Published results for the comparison (E.v), as read from the papers

Compare only like with like, and state the protocol difference.

| Source | Protocol | Reported result |
|---|---|---|
| Pınar et al. (10), ConvNeXt-T | HAM10000, lesion-grouped 5-fold CV, temperature scaled | Accuracy 0.770, balanced accuracy 0.700 [0.674, 0.714], macro-F1 0.640, AUC 0.950, ECE 0.0774 before and 0.0274 after temperature scaling. Six architectures: balanced accuracy 0.664 to 0.700, AUC 0.933 to 0.950. |
| Pınar et al. (10), always-NV baseline | Same | Accuracy 0.669, balanced accuracy 0.143, macro-F1 0.115. This matches our `always_nv` row, which is a good consistency check. |
| Pınar et al. (10), external | BCN20000, no adaptation | Balanced accuracy 0.401 to 0.476, a drop of about 25 points versus internal. |
| Mondal et al. (11), ConvNeXt-Tiny | Lesion-disjoint HAM10000, one split, 1,002-image test fold | Accuracy 78.64%, macro-F1 0.6857, balanced accuracy 0.7460. On external ISIC 2019: accuracy 53.83%. |
| Carse et al. (7)/McKenna and Carse (13), EfficientNet-B7 | ISIC 2019 (not HAM10000), fixed 60/20/20 split, lesion grouping not stated in the text I read | Binary malignant vs benign: balanced accuracy 88.0%, AUC 0.944. Multi-class: balanced accuracy 75.1%, AUC 0.974. |

Differences to explain in the report: our test fold is one lesion-grouped split (about 1,431 images) rather than 5-fold CV; our models are CPU-sized to meet R5; ISIC 2019 results are on a different dataset with a different class set.

## 7. Measurable targets (A.ii), proposed, to be revised after Design A

| ID | Measurable target | How measured |
|---|---|---|
| R1 | Referral rate cap 30%. Referral sensitivity target 0.85 at that cap. If missed, report the best value and why. | Chosen on validation; reported once on test. |
| R2 | Cost ratio r in {2, 5, 10, 20}. Final design must be no worse than the alternatives at all four, or the exception is stated. | Expected cost per lesion. |
| R3 | ECE at most 0.05 on validation after calibration. Risk-coverage curve plotted. Deferral rule reported with cases deferred and errors avoided. | Reliability diagram, ECE, risk-coverage. |
| R4 | Metadata missing at 0%, 25%, 50%, 100%. Referral sensitivity drop at 50% missing at most 0.05. | Masked-metadata runs. |
| R5 | At most 1 s per lesion on CPU. Model file at most 100 MB. Report parameters, training time, hardware. | Timed on your laptop. |
| R6 | Lesion-grouped split, train-only preprocessing, validation-only tuning, one test evaluation. | Asserts in scripts 01 and 02 (done) and the hash check. |

**Intended users and limits (A.i):** a hospital screening team triaging pigmented lesions before dermatologist review. Educational research prototype only, not a medical device. Trained on two sites (Vienna and Australia), so skin-type and acquisition bias are recorded limits. No reader study was done.

## 8. Papers I opened but would not cite

Preprints on missing-metadata fusion (arXiv 2610.03991), cascade triage with a tunable threshold (arXiv 2606.13135) and an image-level HAM10000 split with no lesion grouping (arXiv 2405.14438, LoRA-Ensemble) are relevant background, but I did not open their pages to confirm authors. Leave them out of the reference list unless you check them.
