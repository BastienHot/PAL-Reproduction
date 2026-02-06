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
> pip install transformers==4.9.2 tokenizers==0.10.3 pytorch-lightning==1.5.10 pandas gensim==3.8.3 nltk==3.5 scikit-learn==0.24.1 scipy==1.5.4 statsmodels==0.12.2 tqdm==4.54.0
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
wget https://nlp.stanford.edu/data/glove.6B.zip -O glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt
python -m gensim.scripts.glove2word2vec --input=glove.6B.300d.txt --output=codes/metric/word2vec/glove.6B.300d.model.bin
rm glove.6B.zip glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt
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

### 6. Run inference

Edit `codes/RUN/infer_strat.sh` to set the `CHECKPOINT` variable to your
trained checkpoint path, then:

```bash
bash RUN/infer_strat.sh
```

This produces `gen.json` and `gen.txt` with generated responses and automatic
metrics (BLEU, ROUGE-L, Distinct, etc.).

### 7. Additional evaluation

**EAD score:**
```bash
python get_EAD_score.py --input_file <path_to_gen.json>
```

**Cosine similarity (persona-response alignment):**
```bash
bash RUN/get_gen_sim_cos.sh  # edit INPUT_FILE variable first
```

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

## Reproducing Paper Results (Table 3)

The paper reports these automatic metrics for PAL:

| Model | ACC | PPL | B-2 | B-4 | D-1 | D-2 | E-1 | E-2 | R-L | Cos-Sim |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|---------|
| Blenderbot-Joint | 27.72 | 18.11 | 5.57 | 1.93 | 3.74 | 20.66 | 4.23 | 20.28 | 16.36 | 0.184 |
| PAL (α=0) | 34.25 | 15.92 | 9.28 | 2.90 | 4.72 | 25.56 | 5.87 | 33.05 | 18.27 | 0.229 |
| PAL | 34.51 | 15.92 | 8.75 | 2.66 | 5.00 | 30.27 | 6.73 | 41.82 | 18.06 | 0.244 |

**Known discrepancies** (see [CHANGES.md](CHANGES.md)):
- Paper reports lr=2.5e-5 and warmup=100; code uses lr=1.5e-5 and warmup=0
- Checkpoint selection: authors suggest trying later epochs, not just lowest-loss

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
