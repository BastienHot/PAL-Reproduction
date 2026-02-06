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
- Replaced hardcoded checkpoint path with `$CHECKPOINT` variable (user must set)
- Replaced hardcoded input path with `./_reformat/test.txt`
- Replaced `CUDA_VISIBLE_DEVICES=4` with `CUDA_VISIBLE_DEVICES=0`

### `codes/RUN/interact_strat.sh`
- Replaced hardcoded PAL checkpoint path with `$PAL_CHECKPOINT` variable
- Replaced hardcoded persona extractor checkpoint path with `$PERSONA_CHECKPOINT` variable
- Replaced `CUDA_VISIBLE_DEVICES=5` with `CUDA_VISIBLE_DEVICES=0`

### `codes/RUN/get_gen_sim_cos.sh`
- Replaced hardcoded input path with `$INPUT_FILE` variable

### `codes/RUN/prepare_vanilla.sh`, `train_vanilla.sh`, `infer_vanilla.sh`, `interact_vanilla.sh`
- Rewritten with relative paths and `$CHECKPOINT` variable (based on ESConv originals)
- Replaced `CUDA_VISIBLE_DEVICES` values with `0`

### `codes/get_EAD_score.py`
- Added `argparse` with `--input_file` argument
- Removed hardcoded path `/home/chengjiale/emotion/MISC/generated_data/all_loss/hyp_strategy.json`

### `persona_extractor/train_bart.py`
- Removed dead import `from utils import top_k_top_p_filtering` (the `utils.py` file was never uploaded to the PAL repo — see GitHub issue #8; the function is never called in this file)
- Changed `default_root_dir` from `/home/chengjiale/emotion/Persona_extractor/pl_root` to `./pl_root`
- Changed `hparams.train_dir` from `/home/chengjiale/emotion/Persona_extractor/data/both_original` to `./data`

### `persona_extractor/process_bart_df.py`
- Changed input path from `Persona_chat/personachat/{name}_both_original.txt` to `./data/{name}_both_original.txt`
- Changed output CSV path from `./{name}.csv` to `./data/{name}.csv` (to match `train_bart.py`'s expected data directory)

### `codes/inputters/inputter_utils.py`
- **Bug fix**: Changed data path suffix from `_persona_attention_final` to `_persona_attention_final_rebuttal` (lines 62, 66) to match `prepare.py` and `train.py`. Without this fix, training crashes with `FileNotFoundError` because `prepare.py` writes to `_rebuttal` but `inputter_utils.py` was reading from the old path.

### `codes/models/__init__.py`
- Removed imports and registry entries for `strat_dialogpt` and `vanilla_dialogpt` (files were removed; leaving the imports would crash on `import models`)

---

## Known Discrepancies Between Paper and Code

| Aspect | Paper (Section 4.3) | Code (`RUN/train_strat.sh`) | Status |
|--------|---------------------|----------------------------|--------|
| Learning rate | 2.5e-5 | 1.5e-5 | Unresolved (GitHub issue #11, no author response) |
| Warmup steps | 100 | 0 | Unresolved (same issue) |
| Data split | 7:2:1 (train:valid:test) | 80:10+10:10 with doubled validation | Code uses `valid = data[:10%] + data[20%:30%]` |
| Best epoch | Lowest validation loss | Authors suggest trying later epochs (GitHub issue #6) | Later epochs may give better DISTINCT scores |

---

## Unchanged Files (from PAL repo)

All other Python source files are copied verbatim from the PAL repository:
- `codes/prepare.py`, `train.py`, `infer.py`, `interact.py`
- `codes/get_cos_similarity.py`
- `codes/_reformat/process.py`
- `codes/inputters/*` (all files)
- `codes/models/*` (strat_blenderbot_small.py, strat_blenderbot_small_no_persona.py, vanilla_blenderbot_small.py, model_utils.py, PARAMS.py, __init__.py)
- `codes/utils/*` (all files)
- `codes/metric/*` (all files)
- `codes/apex/*` (all files)
- `codes/CONFIG/strat.json`, `codes/CONFIG/strat_no_persona.json`
- `codes/Blenderbot_small-90M/*` (tokenizer files only; model weights must be downloaded)
