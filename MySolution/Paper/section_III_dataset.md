# III. Dataset and Features

All experiments are conducted on the Trustworthy Intent in Speech (TIS) Corpus released by Maltezou-Papastylianou, Scherer and Paulmann [1]. Section III-A summarises the corpus contents and recording protocol. Section III-B describes the acoustic feature extraction pipeline. Section III-C documents the pre-processing decisions that turn the released CSV files into the standardised numerical arrays consumed by every classifier in Section IV.

## III-A The TIS Corpus

The TIS Corpus is, to our knowledge, the first publicly available speech-trust dataset that systematically spans multiple ethnic backgrounds, two age groups and both sexes. It contains **1,152 audio utterances** from **96 untrained adult speakers**: 40 White, 28 Black and 28 South Asian, recruited via the Prolific platform and in-person word-of-mouth in the United Kingdom [1]. Each speaker contributes 12 utterances, with each utterance recorded **twice** — once in the speaker's natural voice (labelled *Neutral*) and once with the speaker's own attempt to convey trustworthiness (labelled *Trustworthy*) — yielding 576 Neutral and 576 Trustworthy WAV files. The twenty sentences spoken were designed to be semantically neutral with controlled length (7 syllables each), e.g., "*I can drive you if you want*" and "*You may borrow these two books.*"

All recordings were captured online on speakers' own equipment, standardised in Audacity to 48 kHz / 16-bit mono, segmented in Praat [14], normalised to 67 dB, and stored as uncompressed WAV files. The corpus is distributed under the CC-BY 4.0 license on the Open Science Framework at `doi.org/10.17605/OSF.IO/45D8J` [17].

**Table 1** reproduces the speaker demographics from Table 2 of the source paper. The corpus is roughly balanced across the four younger-adult cells (N=20 each for White, Black, South Asian; female/male roughly 50/50), but the older-adult cells are markedly smaller and uneven: only **N=8 older Black** and **N=8 older South Asian** speakers participated, compared to N=20 older White. **Fig. 2** visualises this imbalance and flags the small cells in red. The limited sample size in those cells is a recognised limitation of the corpus and motivates our use of stratified evaluation (Section V-C) and seed-averaged metrics (Section V-A) rather than single-run point estimates.

**Table 1: Speaker demographics in the TIS Corpus, reproduced from Maltezou-Papastylianou et al. [1, Table 2].**

| Ethnicity   | Age-group | Female N | Male N | Total |
|------------:|----------:|---------:|-------:|------:|
| White       | Younger   |    10    |   10   |   20  |
| White       | Older     |    10    |   10   |   20  |
| Black       | Younger   |    11    |    9   |   20  |
| Black       | Older     |     5    |    3   |    8  |
| South Asian | Younger   |    10    |   10   |   20  |
| South Asian | Older     |     4    |    4   |    8  |
| **Total**   |           |   **50** | **46** | **96**|

## III-B Acoustic Feature Extraction

The source paper extracts acoustic and spectral features using the **VoiceLab** software of Feinberg and Cook [15], [16], which provides reproducible Praat-based measurements at the per-utterance level. We use the same released feature CSV (`Speech_dataset_characteristics.csv`) without modification, taking **all 60 numeric acoustic columns** as input to every classifier. The features cluster into five families:

* **Pitch family** (10 features): mean, median, standard deviation, minimum, maximum and floor/ceiling of fundamental frequency (F0), plus four subharmonic-to-harmonic ratio variants.
* **Voice-quality family** (12 features): harmonics-to-noise ratio (HNR), three jitter variants (local, RAP, ppq5, ddp), seven shimmer variants (local, localdb, APQ3, APQ5, APQ11, dda), and cepstral peak prominence (CPP).
* **Formant family** (14 features): means and medians of formants F1 through F4, formant dispersion, average and geometric mean of formants, two estimates of vocal-tract length, and PCA-based formant position.
* **Spectral family** (8 features): mean intensity, spectral tilt, mean energy (two estimators), centre of gravity, kurtosis, skewness, and band-energy difference.
* **Long-term-spectrum family** (6 features): LTAS mean, slope, local peak height, standard deviation, spectral-tilt slope, and intercept.

The source paper [1] analyses approximately ten of these features in detail (the headline ones from Table 1 of that paper) but trains its classifiers on the same broader set we use. We retain all 60 because deep models benefit from redundant correlated features and can learn to suppress uninformative ones during training, whereas hand-pruning would constrain the very interactions the deep models are meant to discover.

**Fig. 3** shows the distributions of six headline features (F0 mean, F0 SD, HNR, Jitter RAP, Shimmer APQ3, CPP) split by intent label. F0 mean and F0 SD show the clearest separation between Neutral and Trustworthy speech — Trustworthy utterances tend to have higher mean pitch and higher pitch variability — replicating the source paper's observation that prosodic *range* drives the trust impression more than absolute pitch height. The four voice-quality features (HNR, jitter, shimmer, CPP) overlap heavily; their contribution to classification is non-linear and emerges only in models that can combine them with the pitch features, which is what motivates the deep architectures of Section IV.

## III-C Pre-Processing

The pipeline is intentionally minimal and identical across all five models, so any performance difference can be attributed to the model itself.

**Label encoding.** `Speaker_Intent` is mapped to a binary target using `sklearn.preprocessing.LabelEncoder` (Neutral → 0, Trustworthy → 1). For DANN-Trust, `Speaker_Ethnicity` is independently encoded to a 3-class integer label using a second `LabelEncoder` instance.

**Missing values.** Any `±inf` values produced by VoiceLab (e.g., division by zero in voice-quality calculations) are first replaced with `NaN`, and all remaining `NaN` values are imputed with the column mean.

**Scaling.** All 60 acoustic features are standardised with `sklearn.preprocessing.StandardScaler`, fit on the training subset only and applied to both train and test. This standardisation is essential for the linear baselines (Logistic Regression is scale-sensitive) and beneficial for the deep models (faster convergence of Adam under unit-variance inputs).

**Stratification.** We split the corpus 80/20 into train (921 utterances) and held-out test (231 utterances) using `train_test_split` with `random_state = 42` and a **joint stratification key** that combines intent and ethnicity into six cells $(2 \text{ intents} \times 3 \text{ ethnicities})$. Stratifying on intent alone — as the source paper effectively does under leave-one-speaker-out cross-validation — lets the test set's ethnicity composition drift, which we observed in pilot experiments to inflate per-ethnicity accuracy variance by a factor of two to three. Joint stratification stabilises per-ethnicity estimates and is the reason the gap standard deviations reported in Section V-I are interpretable.

For comparability with the source paper, we also report leave-one-speaker-out (LOSO) cross-validation results in Section V-B where space permits; LOSO is the harder of the two protocols and typically reduces all reported accuracies by approximately 1 to 3 percentage points.
