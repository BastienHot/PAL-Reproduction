# Changes from Original Repositories

This document tracks every change made from the original codebases:
- **PAL**: https://github.com/chengjl19/PAL
- **ESConv**: https://github.com/thu-coai/Emotional-Support-Conversation (`codes_zcj/`)

---

## Structural Changes

### New files (not in either original repo)
| File | Purpose |
|------|---------|
| `README.md` | Setup guide, environment instructions, reproduction steps |
| `CHANGES.md` | This file — tracks every change from originals |
| `DOWNLOADS.md` | Lists all external downloads with URLs and placement instructions |

### Files integrated from ESConv (not present in PAL repo)
| File | Source | Reason |
|------|--------|--------|
| `codes/_reformat/strategy.json` | ESConv root `strategy.json` | PAL's `_reformat/process.py` requires it but it was missing from the PAL repo |
| `env.yml` | ESConv `codes_zcj/env.yml` | PAL repo had no environment specification (confirmed in GitHub issue #7) |

### Files removed (present in PAL but dropped)
| File | Reason |
|------|--------|
| `models/strat_dialogpt.py` | PAL never uses DialoGPT; carried over from ESConv but unused |
| `models/vanilla_dialogpt.py` | Same as above |

### Files removed (present in ESConv only, not needed for PAL)
| File | Reason |
|------|--------|
| `CONFIG/strat_dialogpt.json` | DialoGPT not used by PAL |
| `CONFIG/vanilla_dialogpt.json` | DialoGPT not used by PAL |
| `DialoGPT-small/` | DialoGPT model files not used by PAL |
| `RUN/*_dialogpt.sh` (6 scripts) | DialoGPT scripts not used by PAL |

---

## Code Changes

### `env.yml` (new, based on ESConv)
- Added `pytorch-lightning==1.5.10` (required by `persona_extractor/train_bart.py`)
- Added `pandas` (required by persona extractor data processing)
- Added `psutil` (required by METEOR metric's memory check in `metric/pycocoevalcap/meteor/meteor.py`)
- Replaced Tsinghua mirror channels with `pytorch` + `conda-forge` for international users
- Renamed environment from `cuda` to `pal`

### `codes/RUN/prepare_strat.sh`
- Replaced hardcoded path `/home/chengjiale/emotion/ESC/Emotional-Support-Conversation/codes_cjl/prepare_data_final/train.txt` with relative `./_reformat/train.txt`
- Replaced `CUDA_VISIBLE_DEVICES=5` with `CUDA_VISIBLE_DEVICES=0`

### `codes/RUN/train_strat.sh`
- Replaced hardcoded path `./prepare_data_final/valid.txt` with `./_reformat/valid.txt`
- Replaced `CUDA_VISIBLE_DEVICES=6` with `CUDA_VISIBLE_DEVICES=0`
- Added comment noting lr/warmup discrepancy between paper and code (GitHub issue #11)

### `codes/RUN/infer_strat.sh`
- Auto-detects latest training run directory (most recent under `DATA/strat.strat_persona_attention_final_rebuttal/`)
- Auto-selects best epoch by parsing `eval_log.csv` for lowest validation loss
- Accepts optional overrides: `bash infer_strat.sh [EPOCH] [RUN_DIR]`
- Replaced hardcoded input path with `./_reformat/test.txt`
- Replaced `CUDA_VISIBLE_DEVICES=4` with `CUDA_VISIBLE_DEVICES=0`

### `codes/RUN/interact_strat.sh`
- Same auto-detection logic as `infer_strat.sh` for PAL checkpoint
- Persona extractor checkpoint path still requires manual configuration (set `PERSONA_CHECKPOINT`)
- Replaced `CUDA_VISIBLE_DEVICES=5` with `CUDA_VISIBLE_DEVICES=0`

### `codes/RUN/get_gen_sim_cos.sh`
- Auto-detects most recent `gen.json` if no argument provided
- Accepts optional explicit path: `bash get_gen_sim_cos.sh [path_to_gen.json]`

### `codes/RUN/infer_vanilla.sh`
- Auto-detects latest run directory under `DATA/vanilla.vanilla/`
- Accepts optional run dir name: `bash infer_vanilla.sh [RUN_DIR]`
- Replaced `CUDA_VISIBLE_DEVICES` with `0`

### `codes/RUN/prepare_vanilla.sh`, `train_vanilla.sh`, `interact_vanilla.sh`
- Rewritten with relative paths (based on ESConv originals)
- Replaced `CUDA_VISIBLE_DEVICES` values with `0`

### `codes/get_EAD_score.py`
- Added `argparse` with `--input_file` argument
- Removed hardcoded path `/home/chengjiale/emotion/MISC/generated_data/all_loss/hyp_strategy.json`
- **Bug fix**: Original code passed entire dict objects to `sent_tokenize` instead of just the `generation` field, producing wildly incorrect EAD scores (~0.37 instead of ~0.05). Fixed by extracting `i['generation']` from each data entry.
- Added E-2 computation (original only computed E-1)

### `codes/get_cos_similarity.py`
- Added `--simcse_model` argument defaulting to `princeton-nlp/sup-simcse-bert-base-uncased` (auto-downloads from HuggingFace)
- Removed hardcoded `./simcse-bert-base-uncased` local path

### `codes/metric/word2vec/evaluate.py`
- **Bug fix**: Replaced hardcoded GloVe path `/home/chengjiale/...` with `os.path.dirname(os.path.abspath(__file__))` (loads from script's own directory)
- **Compatibility fix**: Uses `self.m.vocab[key].index` (gensim 3.x) instead of `self.m.key_to_index[key]` (gensim 4.x), matching the `gensim==3.8.3` version in `env.yml`

### `codes/metric/word2vec/generate_w2v_files.py`
- Replaced hardcoded GloVe source path `/home/zhengchujie/wordvector/english/glove6B` with auto-detection of script directory
- Accepts optional CLI argument for custom path: `python generate_w2v_files.py [path]`

### `codes/inputters/inputter_utils.py`
- **Bug fix**: Changed data path suffix from `_persona_attention_final` to `_persona_attention_final_rebuttal` (lines 62, 66) to match `prepare.py` and `train.py`. Without this fix, training crashes with `FileNotFoundError` because `prepare.py` writes to `_rebuttal` but `inputter_utils.py` was reading from the old path.

### `persona_extractor/train_bart.py`
- Removed dead import `from utils import top_k_top_p_filtering` (the `utils.py` file was never uploaded to the PAL repo — see GitHub issue #8; the function is never called in this file)
- Changed `default_root_dir` from `/home/chengjiale/emotion/Persona_extractor/pl_root` to `./pl_root`
- Changed `hparams.train_dir` from `/home/chengjiale/emotion/Persona_extractor/data/both_original` to `./data`
- Changed `os.environ['CUDA_VISIBLE_DEVICES'] = '2'` to `os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')` (uses `setdefault` so users can override via environment)

### `persona_extractor/process_bart_df.py`
- Changed input path from `Persona_chat/personachat/{name}_both_original.txt` to `./data/{name}_both_original.txt`
- Changed output CSV path from `./{name}.csv` to `./data/{name}.csv` (to match `train_bart.py`'s expected data directory)

### `codes/models/__init__.py`
- Removed imports and registry entries for `strat_dialogpt` and `vanilla_dialogpt` (files were removed; leaving the imports would crash on `import models`)

---

## Code Cleanup (debug prints, commented-out code, Chinese comments)

These changes remove debugging artifacts and improve code readability. No functional behavior is changed.

### `codes/utils/building_utils.py`
- Added `warnings.filterwarnings` to suppress `FutureWarning` from transformers' internal `torch.load(weights_only=False)` — the warning comes from transformers 4.9.2 internals and cannot be fixed at source
- Added `map_location=torch.device('cpu')` to second `torch.load` call in `load_model()`
- Removed commented-out `# print(model.state_dict().keys())`

### `codes/models/strat_blenderbot_small.py` (main PAL model)
- Removed `print(self.strategy_alpha)` active debug print
- Removed ~40 lines of commented-out code blocks (alternative attention, two-part loss experiments)
- Translated Chinese comments to English ("归一化权重" → "Normalize weights via softmax")
- Removed TODO comments and debug print lines

### `codes/models/strat_blenderbot_small_no_persona.py` (ablation model)
- Removed ~50 lines of commented-out code blocks (alternative attention, two-part loss, inference alternatives)
- Translated Chinese comments to English
- Removed TODO comments and debug print lines

### `codes/infer.py`
- Removed ~8 commented-out print lines throughout

### `codes/train.py`
- Removed 3 commented-out debug prints (`# print(global_step, ...)`, `# print(1111)`, `# print(2222)`)

### `codes/prepare.py`
- Removed `# print(processed_data[0].labels)`

### `codes/interact.py`
- Removed `print(inputters)` active debug print
- Removed `print(inputs[0])` active debug print
- Removed 2 commented-out print lines

### `codes/inputters/strat.py`
- Removed ~10 commented-out print lines throughout (debug prints for context, strat, persona, sample_ids)

### `codes/inputters/strat_interact.py`
- Removed `print("Truncated input：", ...)` active debug print (note: contained Chinese punctuation)
- Removed `print("end generate, ", ...)` active debug print with timing
- Removed unused `begin_time` variable and `import time` (only used by removed prints)
- Removed ~8 commented-out print/process lines throughout

### `codes/inputters/train_bart.py` (inside `codes/inputters/`)
- Removed `print(hparams)` active debug print
- Removed 5 commented-out print lines
- Removed commented-out alternative hparams values (freeze_encoder, model alternatives)

### `codes/inputters/inputter_utils.py`
- Removed debug `print(f'./DATA/...')` and 3 commented-out alternative data paths

### `persona_extractor/train_bart.py`
- Removed `print(hparams)` and `print(len(context))` active debug prints
- Removed ~6 commented-out print lines
- Removed ~16 lines of commented-out ESC processing block
- Removed unused variables (`context_number`, `dialog_id`, `now_id`)

---

## Known Discrepancies Between Paper and Code

| Aspect | Paper (Section 4.3) | Code (`RUN/train_strat.sh`) | Status |
|--------|---------------------|----------------------------|--------|
| Learning rate | 2.5e-5 | 1.5e-5 | Code value (1.5e-5) gave best results; paper value (2.5e-5) overfits faster |
| Warmup steps | 100 | 0 | Code value (0) gave best results; warmup had no benefit |
| Data split | 7:2:1 (train:valid:test) | 70:20:10 (= 7:2:1) | No discrepancy — code concatenates two 10% chunks for validation, yielding 910:260:130 samples |
| Best epoch | Lowest validation loss | Authors suggest trying later epochs (GitHub issue #6) | Later epochs may give better DISTINCT scores |

### Hyperparameter Experiment Results

We tested three configurations to understand the lr/warmup discrepancy:

| Config | lr | warmup | ACC | PPL | B-2 | D-2 | Notes |
|--------|----|--------|-----|-----|-----|-----|-------|
| Code defaults | 1.5e-5 | 0 | 32.98 | 15.55 | 8.63 | 19.84 | Best overall |
| Paper values | 2.5e-5 | 100 | 29.57 | 15.24 | 7.42 | 18.53 | Overfits faster, lower ACC |
| Lower lr | 1e-5 | 100 | 28.30 | 21.77 | 6.42 | 18.37 | Underfits (PPL too high) |

---

## Unchanged Files (from PAL repo)

The following files are copied verbatim from the PAL repository with no modifications:
- `codes/_reformat/process.py`
- `codes/inputters/PARAMS.py`
- `codes/models/vanilla_blenderbot_small.py`, `model_utils.py`, `PARAMS.py`
- `codes/utils/eval_utils.py`, `distributed.py`
- `codes/metric/*` (except `word2vec/evaluate.py` and `word2vec/generate_w2v_files.py` noted above)
- `codes/apex/*` (all files — external library, not modified)
- `codes/CONFIG/strat.json`, `codes/CONFIG/strat_no_persona.json`
- `codes/Blenderbot_small-90M/*` (tokenizer files only; model weights must be downloaded)
