# PAL: Persona-Augmented Emotional Support — Reproduction

Clean, self-contained reproduction of **PAL: Persona-Augmented Emotional Support
Conversation Generation** (Cheng et al., ACL 2023 Findings).

This repository integrates code from two sources into a single runnable project:
- [PAL](https://github.com/chengjl19/PAL) — the persona-augmented model
- [Emotional-Support-Conversation](https://github.com/thu-coai/Emotional-Support-Conversation)
  (`codes_zcj/`) — the ESConv baseline framework (Liu et al., ACL 2021)

All changes from the originals are documented in [CHANGES.md](CHANGES.md).
All external downloads are listed in [DOWNLOADS.md](DOWNLOADS.md).

---

## Quick Start

### 1. Create the conda environment

```bash
conda env create -f env.yml
conda activate pal
```

> **Troubleshooting**: The environment pins `cudatoolkit=10.1.243` and
> `pytorch=1.7.0`. If your GPU driver does not support CUDA 10.1, you may need
> to adjust these versions. Check your driver compatibility with
> `nvidia-smi`.

> **Windows users**: The `env.yml` was originally designed for Linux. On Windows
> you may need to remove the `cudatoolkit` line and install PyTorch separately:
> ```bash
> conda create -n pal python=3.8.5
> conda activate pal
> pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
> pip install transformers==4.9.2 tokenizers==0.10.3 pytorch-lightning==1.5.10 pandas gensim==3.8.3 nltk==3.5 scikit-learn==0.24.1 scipy==1.5.4 statsmodels==0.12.2 tqdm==4.54.0 psutil
> ```

### 2. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt')"
```

### 3. Download model weights

**BlenderBot-small-90M** (required — PAL base model):
```bash
# From the repository root:
wget https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin \
     -O codes/Blenderbot_small-90M/pytorch_model.bin
```

**BART-large-CNN** (required — persona extractor base model):
```bash
# transformers should auto-download this, but if it fails:
mkdir -p persona_extractor/bart-large-cnn
cd persona_extractor/bart-large-cnn
wget https://huggingface.co/facebook/bart-large-cnn/resolve/main/pytorch_model.bin -O pytorch_model.bin
wget https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json -O config.json
wget https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json -O tokenizer.json
wget https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json -O vocab.json
wget https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt -O merges.txt
cd ../..
```
If downloaded manually, update `hparams.model_dir_or_name` in
`persona_extractor/train_bart.py` from `"facebook/bart-large-cnn"` to
`"./bart-large-cnn"`.

**GloVe embeddings** (required for full evaluation):
```bash
# Download and extract
wget https://nlp.stanford.edu/data/glove.6B.zip -O glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt
cp glove.6B.300d.txt codes/metric/word2vec/

# Convert to gensim binary format
cd codes/metric/word2vec
python generate_w2v_files.py
cd ../../..

# Clean up source files
rm glove.6B.zip glove.6B.300d.txt
```

See [DOWNLOADS.md](DOWNLOADS.md) for the complete list of external resources
including optional ones (SimCSE, PersonaChat dataset).

### 4. Prepare data

The persona-augmented dataset (PESConv.json) is already included. To generate
the train/valid/test splits:

```bash
cd codes/_reformat
python process.py --add_persona True
cd ..
```

This produces `train.txt`, `valid.txt`, `test.txt` in `codes/_reformat/`.

Then prepare the tokenized features:

```bash
# From codes/ directory:
bash RUN/prepare_strat.sh
```

### 5. Train the PAL model

```bash
# From codes/ directory:
bash RUN/train_strat.sh
```

Checkpoints are saved under `codes/DATA/strat.strat_persona_attention_final_rebuttal/`.
Training creates a timestamped directory (e.g., `2026-0210174049.1.5e-05.4.1gpu`).

### 6. Run inference

The inference script auto-detects the latest training run directory and selects
the best epoch (lowest validation loss from `eval_log.csv`):

```bash
# From codes/ directory:
bash RUN/infer_strat.sh                # auto-detect latest run + best epoch
bash RUN/infer_strat.sh 4              # auto-detect latest run, use epoch 4
bash RUN/infer_strat.sh 4 MY_RUN_DIR   # use specific run dir + epoch
```

This produces `gen.json` and `gen.txt` under a `res_...` subdirectory of your
run directory, with generated responses and automatic metrics (BLEU, ROUGE-L,
Distinct, etc.).

### 7. Additional evaluation

**EAD score** (Expectancy-Adjusted Distinct):
```bash
# Find gen.json in the res_... subdirectory created by inference
python get_EAD_score.py --input_file ./DATA/strat.strat_persona_attention_final_rebuttal/<run_dir>/res_.../gen.json
```

**Cosine similarity** (persona-response alignment via SimCSE):
```bash
bash RUN/get_gen_sim_cos.sh            # auto-detects the latest gen.json
bash RUN/get_gen_sim_cos.sh path/to/gen.json   # or specify explicitly
```

The SimCSE model (`princeton-nlp/sup-simcse-bert-base-uncased`) is downloaded
automatically from HuggingFace. If auto-download fails with
`transformers==4.9.2`, download the model manually and pass the local path:
```bash
python get_cos_similarity.py \
    --input_file <path_to_gen.json> \
    --simcse_model ./simcse-bert-base-uncased
```

---

## Reproduction Results

Results from our reproduction (single GPU, seed=13, lr=1.5e-5, warmup=0, epoch 4)
compared to the paper (Table 3):

| Metric | Paper (PAL) | Reproduction | Notes |
|--------|------------|--------------|-------|
| ACC    | 34.51      | 32.98        | Strategy classification accuracy |
| PPL    | 15.92      | 15.55        | Perplexity (lower is better) |
| B-2    | 8.75       | 8.63         | BLEU-2 |
| B-4    | 2.66       | 2.59         | BLEU-4 |
| D-1    | 5.00       | 3.72         | Distinct-1 |
| D-2    | 30.27      | 19.84        | Distinct-2 |
| E-1    | 6.73       | 4.78         | EAD-1 |
| E-2    | 41.82      | 26.63        | EAD-2 |
| R-L    | 18.06      | 17.73        | ROUGE-L |
| Cos-Sim| 0.244      | 0.235        | SimCSE cosine similarity |

**Analysis**: Most metrics are close to paper values. The main gap is in
diversity metrics (D-1, D-2, E-1, E-2), which are consistently lower. This may
be due to differences in training environment (the paper used multi-GPU
training) or unreported settings. PPL is actually slightly better than the paper.

**Hyperparameter note**: The paper reports lr=2.5e-5 and warmup=100, but the
code defaults to lr=1.5e-5 and warmup=0. We tested both configurations and
found the code defaults produce better overall results. See
[CHANGES.md](CHANGES.md) for details.

---

## Repository Structure

```
PAL-Reproduction/
├── env.yml                          # Conda environment specification
├── README.md                        # This file
├── CHANGES.md                       # All changes from original repos
├── DOWNLOADS.md                     # External downloads with URLs
│
├── persona_extractor/               # BART persona extractor (trained on PersonaChat)
│   ├── process_bart_df.py           #   PersonaChat data preprocessing
│   ├── train_bart.py                #   Training & inference script
│   └── data/                        #   Place PersonaChat .txt files here
│
└── codes/                           # Main PAL model
    ├── _reformat/                   #   Data preprocessing
    │   ├── process.py               #     ESConv/PESConv → train/valid/test splits
    │   ├── strategy.json            #     8 ESC strategy definitions
    │   └── PESConv.json             #     Persona-augmented ESConv dataset
    ├── Blenderbot_small-90M/        #   Tokenizer files (download weights separately)
    ├── CONFIG/                      #   Model configurations
    │   ├── strat.json               #     PAL config (with persona tokens)
    │   └── strat_no_persona.json    #     Ablation config (no persona attention)
    ├── RUN/                         #   Shell scripts (all with relative paths)
    ├── DATA/                        #   Output dir for processed data & checkpoints
    ├── inputters/                   #   Data loading & feature extraction
    ├── models/                      #   Model definitions
    │   ├── strat_blenderbot_small.py          # PAL model (core contribution)
    │   ├── strat_blenderbot_small_no_persona.py  # Ablation (no persona attention)
    │   └── vanilla_blenderbot_small.py        # Vanilla baseline
    ├── metric/                      #   Evaluation metrics (BLEU, ROUGE, Distinct, etc.)
    ├── utils/                       #   Training infrastructure
    ├── apex/                        #   NVIDIA Apex (optional, for FP16)
    ├── prepare.py                   #   Tokenize data → DATA/
    ├── train.py                     #   Training loop
    ├── infer.py                     #   Batch inference + automatic evaluation
    ├── interact.py                  #   Interactive chat demo
    ├── get_EAD_score.py             #   Expectancy-Adjusted Distinct score
    └── get_cos_similarity.py        #   SimCSE persona-response similarity
```

---

## Training the Persona Extractor (optional)

The PESConv.json dataset already contains persona annotations, so this step is
only needed if you want to reproduce the persona extraction process or use the
extractor for interactive mode.

1. Download PersonaChat data (see [DOWNLOADS.md](DOWNLOADS.md)) into
   `persona_extractor/data/`.

2. Preprocess:
   ```bash
   cd persona_extractor
   python process_bart_df.py
   ```

3. Train:
   ```bash
   python train_bart.py
   ```
   Checkpoints are saved under `persona_extractor/pl_root/`.

---

## Citations

If you use this code, please cite both papers:

```bibtex
@inproceedings{cheng2023pal,
  title={PAL: Persona-Augmented Emotional Support Conversation Generation},
  author={Cheng, Jiale and Sabour, Sahand and Sun, Hao and Chen, Zhuang and Huang, Minlie},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  year={2023}
}

@inproceedings{liu2021towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang and Zheng, Chujie and Demasi, Orianna and Sabour, Sahand and Li, Yu and Yu, Zhou and Jiang, Yong and Huang, Minlie},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```
