# Kaggle_LLM_Comp

Code for the Kaggle competition [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
(the Getting Started version of the LMSYS Chatbot Arena human-preference task).

Given a prompt and two LLM responses (A and B), predict the probability that a human preferred A,
preferred B, or called a tie. Scored with multi-class log loss, so calibrated probabilities matter.

## Current stage: baseline (CPU only)

`project/project.py` runs the full pipeline:

1. **Config**: switches paths automatically (`/kaggle/input/...` and `/kaggle/working` on Kaggle, `data/` and `model/` locally), with a fixed seed and a `DEBUG` flag.
2. **Load and parse**: decodes the JSON-list chat columns into plain text. It handles null turns and unparseable cells, and turns the one-hot winner columns into `label` (0 = A, 1 = B, 2 = tie).
3. **EDA**: prints class balance, response lengths per class, how often the longer response wins, and null/empty counts.
4. **Features**: per-text stats (chars, words, newlines, list items, headers, code blocks, bold), A−B differences and log-ratios, TF-IDF cosine similarities (prompt↔A, prompt↔B, A↔B) and an SVD of the A−B TF-IDF difference.
5. **Swap augmentation**: each training row is duplicated with A/B swapped and the labels swapped. Predictions also use swap test-time augmentation (TTA).
6. **Model**: LightGBM multiclass with 5-fold stratified CV, or LogisticRegression if LightGBM is missing. Prints CV log loss next to the uniform (1/3) and class-prior baselines.
7. **Submit**: retrains on all training data, predicts `test.csv`, validates the submission and writes `submission.csv`.

Baseline result (full train, 5-fold CV): **log loss ≈ 1.020** vs **1.0986** for uniform 1/3.

## Layout

```
data/      train.csv, test.csv, sample_submission.csv (from Kaggle; not committed)
model/     outputs: submission.csv, lgbm_baseline.txt (not committed)
project/   project.py (entry point), kernel-metadata.json (for `kaggle kernels push`)
```

## Install (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the data from the competition's Data tab (or `kaggle competitions download -c llm-classification-finetuning`) and unzip it into `data/`.

## Run locally

```powershell
python project/project.py --debug   # 5,000-row sample, ~15 s
python project/project.py           # full run, ~2 min on CPU
```

`DEBUG` can also be turned on with the env var `DEBUG=1` or by setting `DEBUG = True` in the config section.
`USE_TFIDF = False` skips the TF-IDF features for a faster run.
Output goes to `model/submission.csv`.

## Run on Kaggle

The visible `test.csv` has only 3 rows. When you submit, Kaggle reruns the notebook on the hidden test set (~25k rows) with **internet off** and a **9-hour limit**. The code makes no assumptions about the row count.

**Option A: paste into a notebook**

1. Create a new notebook from the competition page ("Code" → "New Notebook"), so the competition data is attached.
2. Paste the whole of `project/project.py` into one cell and run it. Paths switch automatically, and the output goes to `/kaggle/working/submission.csv`.
3. In the notebook's settings panel, turn **Internet off**. Submission is blocked while it is on.
4. Click "Submit to competition".

**Option B: `kaggle kernels push`**

1. Install the CLI (`pip install kaggle`) and put your API token in `%USERPROFILE%\.kaggle\kaggle.json`.
2. In `project/kernel-metadata.json`, replace `YOUR_KAGGLE_USERNAME` with your username. It already sets `enable_internet: false` and attaches the competition data.
3. Push and run it:
   ```powershell
   kaggle kernels push -p project
   kaggle kernels status YOUR_KAGGLE_USERNAME/llm-classification-baseline
   ```
4. Open the kernel on Kaggle, check that it ran and that Internet is off, then click "Submit to competition".

## Next

Stage 2: fine-tune transformers (DeBERTa, then Gemma/Llama with LoRA) on Kaggle GPUs.
