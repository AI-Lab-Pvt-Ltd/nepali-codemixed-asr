# SraVaani fine-tuning for Nepali / Nepali-English code-mixed ASR

This folder fine-tunes **SraVaani** on Nepali speech, including Nepali sentences with English words mixed in. SraVaani is IISc/ARTPARK's FastConformer hybrid TDT-CTC model (~430M parameters, NVIDIA NeMo `EncDecHybridRNNTCTCBPEModel`). It holds:

- the prepared dataset, as NeMo tarred shards (~160 h train, six source datasets),
- `scripts/finetune_nepali.py`, the training script. It is based on the official `finetune.py` and keeps the same NeMo API calls,
- `scripts/evaluate_nepali.py`, which reports test-set WER/CER overall and per source dataset,
- a Python venv with NeMo 2.7.3, PyTorch 2.14 (CUDA 13) and Lightning 2.4, for one RTX 5090 (32 GB).

The base model was trained on Indic text that is single-script, with no code-switching, no punctuation and no digits. Writing English words in Latin script inside Nepali sentences is therefore a **new behaviour** the fine-tune must teach. That is why the defaults train the encoder as well and run for 10 epochs; the official script trains 2 epochs with the encoder frozen.

## Text convention (data and model output)

- Nepali in Devanagari. English words in **lowercase Latin**, as they were spoken: `मैले voxel जे पनि use गर्न सक्छु`.
- **No punctuation and no digits.** Numbers are written out as words. The SraVaani tokenizer (SentencePiece BPE, 5000 pieces) has no digit pieces and almost no punctuation pieces, so the model can never output them.
- The tokenizer is **never changed**. Before training, the script checks that the checkpoint's vocabulary is identical to `assets/sravaani/tokenizer.model` and that every train/validation transcript encodes without `<unk>`.
- The tokenizer applies Unicode NFKC internally. The evaluator therefore normalizes references and hypotheses the same way: NFKC, then whitespace collapsing.

## Folder layout

```
sravaani-finetune/
├── README.md
├── SraVaani-nemo-checkpoint.nemo     # base checkpoint (~1.8 GB, not redistributable; see step 1)
├── .venv/                            # Python 3.12 venv with NeMo
├── assets/
│   ├── sravaani/                     # HF files: tokenizer.model (reference), nemo_model_config.yaml, ...
│   └── sravaani-github/              # official SraVaani repo: finetune.py, resume_finetune_from_ckpt.py, README.md
├── data/                             # the dataset (NeMo tarred format)
│   ├── train/       shard_0000.tar … shard_0031.tar + shard_NNNN.json + manifest.json
│   ├── validation/  shard_0000.tar, shard_0001.tar + jsons + manifest.json
│   └── test/        shard_0000.tar, shard_0001.tar + jsons + manifest.json
├── scripts/
│   ├── finetune_nepali.py
│   └── evaluate_nepali.py
├── experiments/                      # created by training (checkpoints, TensorBoard, final .nemo)
├── reports/                          # created by evaluation (JSONL + markdown tables)
└── work/                             # scratch: sample export, smoke-test logs
```

Shard format: uncompressed tar files whose members are `<id>.wav` (16 kHz mono 16-bit). Each tar has a JSONL manifest `shard_NNNN.json` with its lines in tar order. Each line has `audio_filepath` (the member name inside the tar), `duration`, `text`, `source_dataset` and `shard_id`. `manifest.json` is all the shard manifests concatenated. The scripts find the shards on disk and build the brace pattern themselves (`shard_{0000..0031}.tar`).

## Dataset statistics

manifest_sha256=76e5487a376416148d53efa0e76a33b9565a2216b96875d6bbfa4cd3c3bf24cb  
locked=2026-10-13T16:56:20Z  
train=105448  
validation=5049  
test=4610

| subset | train | validation | test |
|---|---|---|---|
| openslr_nepali_asr | 69,133 / 57.62 h | 1,472 / 1.22 h | 1,500 / 1.25 h |
| saileshbro_nepali_cs_asr | 15,867 / 44.53 h | 2,277 / 6.35 h | 1,574 / 4.25 h |
| bijaya_khanal_cs_asr | 6,305 / 28.47 h | 363 / 1.63 h | 358 / 1.63 h |
| uttam_parajuli_asr | 12,990 / 19.61 h | 816 / 1.29 h | 730 / 1.12 h |
| nep_eng_code_mixed_asr | 570 / 0.99 h | 48 / 0.08 h | 373 / 0.62 h |
| technical_interview_asr | 583 / 1.68 h | 73 / 0.20 h | 75 / 0.21 h |
| **total** | **105,448 / 152.91 h** | **5,049 / 10.77 h** | **4,610 / 9.08 h** |

- Source pool (non-medical): 116,175 clips / 175.2 h; 29.7% had characters the SraVaani tokenizer cannot encode before normalisation.
- Dropped during preparation: 1,068 clips / 2.4 h (ambiguous number reading 644, hypothesis too different 315, label too long for CTC 109).
- Code-mixing: 43.8% of clips contain an English (Latin) word; 24.2% of all words are English. Vocabulary 69,158 word types (14,876 Latin).
- Durations 0.99–28.6 s (mean 5.40 s); training skips the 33 train clips > 20 s.
- Numbers: 17,971 clips contained digits/symbols; 35,157 numeric spans were spelled out from audio evidence (45.1% Nepali, 53.1% English).
- Full reports: `reports/selection_report.md`, `reports/validation_report.md`, `reports/label_audit.md`, `reports/text_normalization.md`.

## Step by step

### 1. Get the base checkpoint

The `.nemo` checkpoint is not on Hugging Face. Request access through the form linked in the official README (`assets/sravaani-github/README.md`):

- Request form: **[Request access here](https://script.google.com/macros/s/AKfycbyAC0vGcGSbu55LNKOn_hY3TsPy8CyF1nasLJULWMrzCdksBJG33bUfLbiKee_m_aKH-w/exec)**
- Google Drive folder (once access is granted): **[Download SraVaani-nemo-checkpoint.nemo](https://drive.google.com/drive/folders/1QdMG0R3oGgOThm7uqYeMnt8zGOSx467x)**

### 2. Put it in the project folder

```
/home/ubuntu/sravaani-finetune/SraVaani-nemo-checkpoint.nemo
```

`finetune_nepali.py` uses this file when `--nemo-checkpoint` is not given. `evaluate_nepali.py` always needs `--nemo-checkpoint`.

### 3. Activate the venv

```bash
cd /home/ubuntu/sravaani-finetune
source .venv/bin/activate
```

### 4. Dry run (about 15 s, no training)

```bash
python scripts/finetune_nepali.py --dry-run
```

The dry run:

- checks the shards against the manifests (counts, tar order, ids, 16 kHz mono, durations) and prints per-split statistics,
- restores the checkpoint on CPU and checks that its tokenizer equals `assets/sravaani/tokenizer.model`,
- encodes every transcript and checks for `<unk>`,
- builds the NeMo tarred train/validation dataloaders exactly as training does, pulls batches, and checks that every sample's tokens and audio length match its manifest line,
- prints the step plan (steps per epoch, warmup, validation interval) and free disk space.

It ends with `[dry-run] PASSED`. Add `--dry-run-no-model` to skip the checkpoint restore, or `--data-dir work/sample_export` to try it on the small sample.

### 5. Train

The recommended settings are the defaults:

```bash
python scripts/finetune_nepali.py 2>&1 | tee experiments/train.log
```

This is equivalent to:

```bash
python scripts/finetune_nepali.py --nemo-checkpoint SraVaani-nemo-checkpoint.nemo \
    --epochs 10 --batch-size 8 --accumulate 8 --lr 1e-4 --encoder-lr-scale 0.5 \
    --warmup-ratio 0.05 --min-lr 1e-6 --weight-decay 1e-3 --precision bf16-mixed \
    --max-duration 20 --val-max-duration 30 --min-duration 0.5 --fused-batch-size 1 \
    --val-check-interval 0.5 --save-top-k 1 --exp-dir experiments/sravaani-nepali-ft
```

| Setting | Default | Why |
|---|---|---|
| encoder | trained, LR × 0.5 | Code-mixed English is a new acoustic and lexical behaviour, so the encoder has to adapt. Half the head LR limits drift from the pretrained representation. `--freeze-encoder` gives the official behaviour. |
| epochs | 10 (~16k optimizer steps) | A new behaviour, not a small domain shift. The best-val_wer checkpoint is exported, so extra epochs cost only time. |
| batch × accumulate | 8 × 8 = 64 utterances (~6 min of audio per step) | Measured to fit 32 GB even for worst-case batches (see the smoke test below). |
| LR / schedule | AdamW 1e-4 peak, cosine to 1e-6, 5% warmup | The official optimizer and schedule. Warmup is longer than the official 10 steps because the encoder is unfrozen and the optimizer starts fresh. |
| weight decay | 1e-3 | Same value as the official script and the checkpoint's config. |
| precision | bf16-mixed | Native on the RTX 5090; no loss scaling needed. |
| max-duration | 20 s train / 30 s validation | Only 37 of ~117k clips are longer than 20 s. Because of padding, one of them makes a batch of 8 run out of memory. Validation keeps every clip. |
| fused-batch-size | 1 (checkpoint has 4) | Computes the TDT joint and loss one utterance at a time. This removes a 4–7 GB memory spike and is no slower. |
| val-check-interval | 0.5 epoch | About 20 val_wer points over the run for picking the best checkpoint. |
| save-top-k | 1, plus last.ckpt | Each `.ckpt` is ~5.4 GB (see Disk below). |

During training you get a line every 50 steps, `[step N] train_loss=… lr=… s/step peak_mem=…GiB`, and one per validation, `[step N] val_wer=… val_wer_ctc=…`. TensorBoard:

```bash
tensorboard --logdir experiments/sravaani-nepali-ft
```

Outputs, all in `experiments/sravaani-nepali-ft/`:

- `sravaani-nepali-best.nemo`: weights of the best val_wer checkpoint. **Use this one.**
- `sravaani-nepali-last.nemo`: the final weights.
- `checkpoints/*.ckpt`: Lightning checkpoints (best + last) for resuming.
- `run_args.json`, logs and TensorBoard events.

Throughput estimate: ~0.11–0.15 s per batch of 8 on the sample data. That works out to roughly 30–40 min per epoch and **5–7 h for 10 epochs**. The real number is in the `s/step` field of the log (one step = 8 batches).

**Optional two-stage recipe.** Use this if the encoder seems to drift, for example if pure-Nepali WER on OpenSLR gets worse:

```bash
# stage 1: heads only, like the official script
python scripts/finetune_nepali.py --freeze-encoder --epochs 2 --exp-dir experiments/stage1
# stage 2: end-to-end, starting from stage 1
python scripts/finetune_nepali.py --nemo-checkpoint experiments/stage1/sravaani-nepali-best.nemo \
    --lr 5e-5 --epochs 8 --exp-dir experiments/stage2
```

### 6. Evaluate

```bash
# base model (zero-shot) and the fine-tuned model, TDT decoding
python scripts/evaluate_nepali.py --nemo-checkpoint SraVaani-nemo-checkpoint.nemo --tag base
python scripts/evaluate_nepali.py --nemo-checkpoint experiments/sravaani-nepali-ft/sravaani-nepali-best.nemo \
    --tag finetuned --decoder both          # TDT and the auxiliary CTC head
```

The evaluator reads the test tars straight into memory and passes numpy arrays to `model.transcribe()`. It writes these files to `reports/`:

- `<tag>_test_<decoder>.jsonl`: one line per clip: id, source_dataset, duration, ref, hyp.
- `<tag>_test.md`: a table of WER %, CER % and EN-WER % (WER over Latin-script English words only), overall and per source_dataset.
- `<tag>_test_metrics.json`: the same numbers as JSON.

Other options: `--split validation`, `--limit 200` for a quick check, `--batch-size`, `--amp` for bf16 inference, and `--input-mode files` (extracts WAVs to a temp dir, a fallback).

### 7. Notes

- **Outputs have no digits or punctuation.** References follow the same convention, so scores are fair. If you need numerals in the final text, add inverse text normalization as a post-processing step.
- **GPU memory.** The TDT loss holds an fp32 tensor of batch × frames × tokens × 5006, so memory grows with clip length × transcript length × batch size. Typical batches use ~14 GiB; the worst case at the defaults is ~26 GiB. If you hit `CUDA out of memory`:
  - First, keep the effective batch and halve the batch size: `--batch-size 4 --accumulate 16`.
  - Next, lower `--max-duration` (for example 15).
  - Make sure no other process is using the GPU (`nvidia-smi`). The defaults assume the card is free.
  - `--max-duration 30` (keeps every clip) needs `--batch-size 4 --accumulate 16`.

  The scripts set `PYTORCH_ALLOC_CONF=expandable_segments:True`. Without it, variable-length batches fragment the allocator: 30.7 GiB reserved for 19.7 GiB used.
- **Disk.** Each `.ckpt` is ~5.4 GB (weights + optimizer state) and each `.nemo` ~1.8 GB. With `--save-top-k 1` a run needs about 20 GB. The script warns if less than 15 GB is free. Delete old `experiments/*/checkpoints` once you have the `.nemo` you want.
- **Resume after an interruption.** Rerun the **same command** with `--resume`:

  ```bash
  python scripts/finetune_nepali.py --resume
  ```

  This restores weights, optimizer, LR scheduler and step from `checkpoints/*-last.ckpt` in `--exp-dir`. The script refuses `--freeze-encoder` / `--encoder-lr-scale` values that differ from the original run, because they change the optimizer layout. A fresh run into an `--exp-dir` that already has checkpoints is refused; pass `--resume`, or choose a new `--exp-dir`. On resume, NeMo moves the previous run's log files into `run_N/` inside the exp dir.
- **Smoke test (done on the real checkpoint, `work/smoke/*.log`).**
  - The dry run passes with the real `restore_from`; the tokenizer is identical.
  - A 40-step run (batch 8, bf16) with a validation pass, then a `--resume` from step 20 to 30, both worked and exported `.nemo` files.
  - After just 30 steps, TDT WER on the 300-clip sample test went from 50.4% (base) to 39.1%, and EN-WER from 78.6% to 53.0%.
  - Peak memory: 14.4 GiB on real batches at batch 8. Synthetic worst-case batches of high-token-rate clips: 8 × 20 s → 26.1 GiB (fits); 4 × 28.6 s → 22.4 GiB; 8 × 28.6 s → out of memory.
- **Environment quirks handled in the scripts.**
  - NumPy 2.5 removed `np.row_stack`, which numba-cuda 0.30 still uses when NeMo compiles its TDT loss kernel. The scripts alias it to `np.vstack`.
  - NeMo's Lhotse dataloader cannot read `shard_NNNN.tar` names, so `use_lhotse=False` is forced.
  - NeMo's tarred loader shuffles shard order only, not the clips inside a shard. The export is pre-shuffled globally, and the dry run reports the longest single-source run.
- **Manifest form.** The default `--manifest-mode combined` uses `<split>/manifest.json`. `--manifest-mode shards` passes the list of shard JSONs instead; both were tested.
