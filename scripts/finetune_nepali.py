#!/usr/bin/env python
"""Fine-tune SraVaani (FastConformer hybrid TDT-CTC, ~430M) on Nepali / Nepali-English
code-mixed speech stored as NeMo tarred shards.

Based on the official SraVaani finetune.py (assets/sravaani-github/finetune.py): same NeMo
API (EncDecHybridRNNTCTCBPEModel.restore_from -> edit cfg under open_dict -> setup_*_data ->
setup_optimization -> pl.Trainer + exp_manager -> fit -> save_to), same tarred-data keys,
same AdamW + CosineAnnealing schedule and max_steps formula, same MetricsLogger output.
The tokenizer is never changed.

Differences from the official script (all deliberate):
  * every constant is a CLI flag, and shard brace patterns are discovered from disk;
  * the encoder is trained by default (at a lower LR than the heads) instead of frozen;
  * the Trainer is attached to the model *before* the dataloaders are built, so NeMo can set
    the correct number of batches per epoch for the iterable tarred dataset;
  * pre-flight checks: the restored tokenizer must equal assets/sravaani/tokenizer.model and
    every train/validation transcript must encode without <unk>;
  * best-by-val_wer and last weights are both exported as .nemo;
  * --dry-run exercises the whole data path (shards -> NeMo tarred dataloader -> batches,
    restoring the checkpoint for the tokenizer check, or --dry-run-no-model without one);
    --resume continues an interrupted run with optimizer state;
  * memory/disk defaults measured on the real checkpoint (see README "Smoke test").

Examples:
  python scripts/finetune_nepali.py --dry-run
  python scripts/finetune_nepali.py --nemo-checkpoint SraVaani-nemo-checkpoint.nemo
  python scripts/finetune_nepali.py --nemo-checkpoint SraVaani-nemo-checkpoint.nemo --resume
"""

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_EXP_DIR = PROJECT_DIR / "experiments" / "sravaani-nepali-ft"
DEFAULT_CHECKPOINT = PROJECT_DIR / "SraVaani-nemo-checkpoint.nemo"
REFERENCE_TOKENIZER = Path(os.environ.get("SRAVAANI_TOKENIZER",
    str(PROJECT_DIR / "assets" / "sravaani" / "tokenizer.model")))
SAMPLE_RATE = 16000
SHARD_RE = re.compile(r"shard_(\d+)\.tar")

# Arguments that change the optimizer's parameter groups. A --resume with different values
# cannot load the saved optimizer state, so we refuse it up front.
RESUME_STRUCTURAL_ARGS = ("freeze_encoder", "encoder_lr_scale")
# Arguments that change the LR schedule / epoch length; differing values only warn.
RESUME_SCHEDULE_ARGS = ("epochs", "max_steps", "batch_size", "accumulate", "lr", "min_lr",
                        "warmup_steps", "warmup_ratio", "weight_decay", "max_duration",
                        "val_max_duration", "min_duration", "manifest_mode")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _interval(value: str):
    """val-check interval: float <= 1.0 = fraction of an epoch, larger integer = batches."""
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v if v <= 1.0 else int(v)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune SraVaani on Nepali / Nepali-English code-mixed tarred data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("paths")
    io.add_argument("--nemo-checkpoint", type=Path, default=None,
                    help="Base SraVaani .nemo or a previous fine-tuned .nemo. Default: "
                         "./SraVaani-nemo-checkpoint.nemo in the project dir, if present.")
    io.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="Folder with train/ and validation/ shard folders.")
    io.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR,
                    help="Run folder: checkpoints, TensorBoard logs and the final .nemo files.")
    io.add_argument("--run-name", default="sravaani-nepali",
                    help="Prefix for checkpoint / .nemo file names.")
    io.add_argument("--manifest-mode", choices=("combined", "shards"), default="combined",
                    help="combined: manifest_filepath=<split>/manifest.json; shards: the list "
                         "of per-shard shard_NNNN.json files. Both describe the same clips.")

    tr = p.add_argument_group("training (defaults tuned for ~160 h on one 32 GB GPU)")
    # ~160 h is roughly 1.6k optimizer steps/epoch at an effective batch of 64. The base
    # model was trained on single-script, no-code-switching text, so emitting Latin-script
    # English inside Nepali is a new behaviour, not a small domain shift: 10 epochs (~16k
    # steps, est. 5-7 h on the 5090) rather than the official 2. The best val_wer checkpoint
    # is exported, so surplus epochs cost time, not accuracy.
    tr.add_argument("--epochs", type=int, default=10)
    # 8 x 8 = effective batch 64 utterances (~6 min of audio per optimizer step). Memory is
    # dominated by the TDT loss, which keeps an fp32 gradient of B x T x U x 5006 until
    # backward (~1.4 GB per 20 s / 240-token clip) on top of ~7 GB of weights + AdamW.
    # Measured on the real checkpoint (RTX 5090, bf16, fused joint 1): typical batches peak
    # at ~14 GiB, a worst-case batch of 8 x 20 s high-token-rate clips at 26 GiB.
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--accumulate", type=int, default=8,
                    help="Gradient accumulation; effective batch = batch-size x accumulate.")
    # Peak LR of the decoder / joint / CTC head. 1e-4 is the official fine-tuning LR.
    tr.add_argument("--lr", type=float, default=1e-4)
    # The official script freezes the encoder, which can only re-weight what the decoder
    # already does; acoustic modelling of English words inside Nepali speech lives in the
    # encoder. So we train it, but at half the head LR to limit drift from the pretrained
    # multilingual representation (discriminative fine-tuning). 1.0 = one LR for all.
    tr.add_argument("--encoder-lr-scale", type=float, default=0.5,
                    help="Encoder LR = lr x this (ignored with --freeze-encoder).")
    tr.add_argument("--min-lr", type=float, default=1e-6,
                    help="Cosine floor (same as the official script).")
    # A fresh AdamW on a pretrained model needs its second-moment estimates warmed up
    # before taking full-size steps; the official 10 steps is too short for an unfrozen
    # encoder. 5% of the run (~800 steps with the defaults) is a standard choice.
    tr.add_argument("--warmup-ratio", type=float, default=0.05,
                    help="Warmup as a fraction of total optimizer steps.")
    tr.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps; overrides --warmup-ratio when > 0.")
    tr.add_argument("--weight-decay", type=float, default=1e-3,
                    help="AdamW weight decay (official / NeMo FastConformer value).")
    enc = tr.add_mutually_exclusive_group()
    enc.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true",
                     help="Official behaviour: train only decoder/joint/CTC head. Useful as "
                          "stage 1 of a two-stage recipe (see README).")
    enc.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false",
                     help="Train end-to-end (default).")
    p.set_defaults(freeze_encoder=False)
    # The RTX 5090 (Blackwell) has native bf16; bf16 autocast needs no loss scaling and
    # roughly halves activation memory compared with fp32.
    tr.add_argument("--precision", default="bf16-mixed",
                    help="Lightning precision: bf16-mixed, 16-mixed or 32.")
    # Padding makes one long clip cost as much as B long clips. Only 37 of ~117k clips are
    # > 20 s (0.24 h), but with them a batch of 8 needs > 32 GB (measured: OOM at 28.6 s even
    # with fused joint 1), so training drops them; validation keeps everything (no loss is
    # computed there, so long clips are cheap).
    tr.add_argument("--max-duration", type=float, default=20.0,
                    help="Drop TRAINING clips longer than this (seconds). 30 keeps every clip "
                         "but then needs --batch-size 4 --accumulate 16.")
    tr.add_argument("--val-max-duration", type=float, default=30.0,
                    help="Drop VALIDATION clips longer than this (seconds); 30 keeps all.")
    tr.add_argument("--min-duration", type=float, default=0.5,
                    help="Drop clips shorter than this (seconds), train and validation.")
    tr.add_argument("--augment-config", type=Path, default=None,
                    help="YAML of NeMo train_ds.augmentor effects (noise, babble, music, room "
                         "echo, phone, speed, gain) applied on the fly to TRAINING clips only, "
                         "e.g. configs/augmentor.yaml. Default: off.")
    tr.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers (capped at the number of shards in the split).")
    # Validating twice per epoch gives ~20 val_wer points over the run to pick the best
    # checkpoint from, at ~2-3% extra time (validation is ~5% of the train set).
    tr.add_argument("--val-check-interval", type=_interval, default=0.5,
                    help="<=1.0: fraction of an epoch; >1: number of training batches.")
    tr.add_argument("--max-steps", type=int, default=-1,
                    help="Stop after this many optimizer steps (also sets the LR schedule "
                         "length). Handy for a short smoke test with the real checkpoint.")
    # The checkpoint ships fuse_loss_wer=True with fused_batch_size=4: the joint + loss run
    # in sub-batches of 4, whose transient fp32 logits (4 x T x U x 5006) are 4-7 GiB for
    # long clips. Sub-batches of 1 cut that peak with no measurable slowdown (0.108 vs
    # 0.112 s per batch of 8).
    tr.add_argument("--fused-batch-size", type=int, default=1,
                    help="TDT joint/loss sub-batch size (NeMo fuse_loss_wer); 0 keeps the "
                         "checkpoint's setting (4).")
    # A full Lightning .ckpt (weights + AdamW state) is ~5.4 GB, a .nemo ~1.8 GB. top-1 +
    # last (needed for --resume) keeps the run within ~20 GB of disk.
    tr.add_argument("--save-top-k", type=int, default=1,
                    help="Checkpoints kept by val_wer (each ~5.4 GB), plus last.ckpt.")
    tr.add_argument("--log-every", type=int, default=50,
                    help="Print train_loss every N optimizer steps.")
    tr.add_argument("--seed", type=int, default=1234)
    tr.add_argument("--allow-subset-manifest", action="store_true",
                    help="Accept shard manifests that list an ordered SUBSET of each tar's members (data "
                         "ablations built by filtering manifest.json over the full shards; NeMo skips "
                         "unlisted members). Default: manifests must match the tars exactly.")

    wb = p.add_argument_group("Weights & Biases")
    wb.add_argument("--wandb", action="store_true",
                    help="Also log to W&B (train loss, val WER, LR, config) next to TensorBoard.")
    wb.add_argument("--wandb-project", default="sravaani-nepali-codemix")
    wb.add_argument("--wandb-name", default=None, help="W&B run name (default: --run-name).")
    wb.add_argument("--wandb-mode", choices=["online", "offline"], default=None,
                    help="Default: online if WANDB_API_KEY is set, else offline (sync later with "
                         "`wandb sync <exp-dir>/wandb/offline-run-*`).")
    wb.add_argument("--wandb-tags", default="", help="Comma-separated tags, e.g. 'unfrozen,main'.")

    mode = p.add_argument_group("modes")
    mode.add_argument("--resume", action="store_true",
                      help="Continue the run in --exp-dir from its last checkpoint (weights, "
                           "optimizer, scheduler and step). Use the same flags as the original run.")
    mode.add_argument("--dry-run", action="store_true",
                      help="Check shards, manifests and tokenizer coverage, restore the "
                           "checkpoint on CPU (tokenizer-equality check), build the NeMo tarred "
                           "train/validation dataloaders exactly as training would and pull a "
                           "few batches; no training. Uses ./SraVaani-nemo-checkpoint.nemo if "
                           "--nemo-checkpoint is not given and it exists.")
    mode.add_argument("--dry-run-no-model", action="store_true",
                      help="With --dry-run: skip the checkpoint restore and use "
                           "assets/sravaani/tokenizer.model directly (fast, data-only check).")
    mode.add_argument("--dry-run-batches", type=int, default=2,
                      help="Batches to pull per split in --dry-run.")
    args = p.parse_args(argv)

    if args.dry_run and args.dry_run_no_model:
        args.nemo_checkpoint = None
    elif args.nemo_checkpoint is None:
        if DEFAULT_CHECKPOINT.exists():
            args.nemo_checkpoint = DEFAULT_CHECKPOINT
        elif not args.dry_run:
            p.error(f"--nemo-checkpoint is required (no {DEFAULT_CHECKPOINT.name} in the project dir)")
    if args.nemo_checkpoint is not None and not args.nemo_checkpoint.is_file():
        p.error(f"checkpoint not found: {args.nemo_checkpoint}")
    if args.batch_size < 1 or args.accumulate < 1 or args.epochs < 1:
        p.error("--batch-size, --accumulate and --epochs must be >= 1")
    if not args.freeze_encoder and args.lr * args.encoder_lr_scale < args.min_lr:
        p.error("encoder LR (lr x encoder-lr-scale) must be >= --min-lr")
    if args.min_duration >= min(args.max_duration, args.val_max_duration):
        p.error("--min-duration must be < --max-duration and --val-max-duration")
    args.data_dir = args.data_dir.resolve()
    args.exp_dir = args.exp_dir.resolve()
    return args


def log(msg=""):
    print(msg, flush=True)


def numpy_compat():
    """NumPy 2.5 removed np.row_stack (an alias of vstack), but numba-cuda 0.30 still
    registers an overload for it when NeMo's TDT/RNNT loss first compiles its CUDA kernel."""
    import numpy as np
    if not hasattr(np, "row_stack"):
        np.row_stack = np.vstack


# --------------------------------------------------------------------------------------
# Data discovery and validation (pure Python; no NeMo needed)
# --------------------------------------------------------------------------------------
def read_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"[data] {path}:{ln}: invalid JSON ({e})")
    return rows


def discover_split(data_dir: Path, split: str, manifest_mode: str) -> dict:
    """Find shard_NNNN.tar (+ paired .json) in data_dir/split and build NeMo paths.

    Returns tar_pattern like '<dir>/shard_{0000..0031}.tar' (NeMo brace-expands it) and
    manifest = '<dir>/manifest.json' (combined) or the list of shard jsons (shards).
    """
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise SystemExit(f"[data] missing split folder: {split_dir}")
    tars = sorted(p for p in split_dir.glob("shard_*.tar") if SHARD_RE.fullmatch(p.name))
    if not tars:
        raise SystemExit(f"[data] no shard_NNNN.tar files in {split_dir}")
    digits = {len(SHARD_RE.fullmatch(p.name).group(1)) for p in tars}
    if len(digits) != 1:
        raise SystemExit(f"[data] mixed shard number widths in {split_dir}: {sorted(digits)}")
    width = digits.pop()
    indices = [int(SHARD_RE.fullmatch(p.name).group(1)) for p in tars]
    if indices != list(range(len(tars))):
        raise SystemExit(f"[data] shard numbers in {split_dir} are not 0..{len(tars) - 1} "
                         f"without gaps: {indices[:10]}...")
    shard_jsons = [p.with_suffix(".json") for p in tars]
    missing = [p.name for p in shard_jsons if not p.is_file()]
    if missing:
        raise SystemExit(f"[data] {split}: shard manifests missing: {missing[:5]}")
    n = len(tars)
    if n == 1:
        tar_pattern = str(tars[0])
    else:
        tar_pattern = str(split_dir / f"shard_{{{0:0{width}d}..{n - 1:0{width}d}}}.tar")
    combined = split_dir / "manifest.json"
    if manifest_mode == "combined":
        if not combined.is_file():
            raise SystemExit(f"[data] {combined} not found (use --manifest-mode shards?)")
        manifest = str(combined)
    else:
        manifest = [str(p) for p in shard_jsons]
    return {"split": split, "dir": split_dir, "n_shards": n, "tars": tars,
            "shard_jsons": shard_jsons, "combined": combined, "tar_pattern": tar_pattern,
            "manifest": manifest}


def _tar_member_names(path: Path):
    with tarfile.open(path, "r:") as tf:
        return [m.name for m in tf if m.isfile()]


def _check_wav_headers(path: Path, rows, limit=64):
    """Check the first `limit` WAVs of a shard: 16 kHz mono, length == manifest duration."""
    import io
    import soundfile as sf
    dur = {r["audio_filepath"]: float(r["duration"]) for r in rows}
    problems = []
    with tarfile.open(path, "r:") as tf:
        for i, m in enumerate(tf):
            if i >= limit:
                break
            if not m.isfile():
                continue
            info = sf.info(io.BytesIO(tf.extractfile(m).read()))
            secs = info.frames / info.samplerate
            if info.samplerate != SAMPLE_RATE or info.channels != 1:
                problems.append(f"{m.name}: {info.samplerate} Hz, {info.channels} ch")
            elif m.name in dur and abs(secs - dur[m.name]) > 0.02:
                problems.append(f"{m.name}: audio {secs:.3f} s vs manifest {dur[m.name]:.3f} s")
    return problems


def _is_subsequence(sub, seq) -> bool:
    """True if `sub` appears in `seq` in the same order (not necessarily contiguous)."""
    it = iter(seq)
    return all(x in it for x in sub)


def validate_split(info: dict, args, max_duration: float) -> dict:
    """Validate manifests against the contract and return stats + the kept rows."""
    split = info["split"]
    shard_rows = [read_jsonl(p) for p in info["shard_jsons"]]
    rows = [r for rs in shard_rows for r in rs]
    if info["combined"].is_file():
        n_comb = sum(1 for line in open(info["combined"], encoding="utf-8") if line.strip())
        if n_comb != len(rows):
            raise SystemExit(f"[data] {split}: manifest.json has {n_comb} lines but the shard "
                             f"manifests have {len(rows)}")
    problems = []
    ids = Counter()
    for r in rows:
        for key in ("audio_filepath", "duration", "text"):
            if key not in r:
                problems.append(f"missing '{key}': {str(r)[:120]}")
                break
        else:
            name = r["audio_filepath"]
            stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
            if "/" in name or ext.lower() != "wav":
                problems.append(f"audio_filepath must be '<id>.wav' inside the tar: {name}")
            elif "." in stem:
                # WebDataset splits member names at the FIRST dot, so 'a.b.wav' would be
                # read as key 'a' and silently never match its manifest line.
                problems.append(f"id contains a dot (breaks WebDataset keys): {name}")
            if not isinstance(r["text"], str) or not r["text"].strip():
                problems.append(f"empty text: {name}")
            try:
                if float(r["duration"]) <= 0:
                    problems.append(f"non-positive duration: {name}")
            except (TypeError, ValueError):
                problems.append(f"bad duration: {name}")
            ids[stem] += 1
    dups = [k for k, c in ids.items() if c > 1]
    if dups:
        problems.append(f"{len(dups)} duplicate ids, e.g. {dups[:3]}")
    # Tar order must equal manifest order (contract); check the first and last shard.
    for idx in sorted({0, info["n_shards"] - 1}):
        names = _tar_member_names(info["tars"][idx])
        expect = [r["audio_filepath"] for r in shard_rows[idx]]
        if names != expect and getattr(args, "allow_subset_manifest", False) and _is_subsequence(expect, names):
            continue  # filtered manifest over full shards: NeMo yields only the listed members
        if names != expect:
            problems.append(f"{info['tars'][idx].name}: tar members ({len(names)}) differ from "
                            f"{info['shard_jsons'][idx].name} entries ({len(expect)}) or their order")
    if not problems:
        problems += _check_wav_headers(info["tars"][0], shard_rows[0])
    if problems:
        log(f"[data] {split}: {len(problems)} problem(s):")
        for msg in problems[:15]:
            log(f"    - {msg}")
        raise SystemExit(f"[data] {split}: fix the manifests/shards above first")

    kept = [r for r in rows if args.min_duration <= float(r["duration"]) <= max_duration]
    stats = {
        "max_duration": max_duration,
        "n_total": len(rows),
        "n_kept": len(kept),
        "n_too_short": sum(float(r["duration"]) < args.min_duration for r in rows),
        "n_too_long": sum(float(r["duration"]) > max_duration for r in rows),
        "hours_kept": sum(float(r["duration"]) for r in kept) / 3600.0,
        "max_dur": max(float(r["duration"]) for r in rows),
        "by_source": Counter(r.get("source_dataset", "?") for r in kept),
        "shard_sizes": [len(rs) for rs in shard_rows],
        "not_nfkc": sum(unicodedata.normalize("NFKC", r["text"]) != r["text"] for r in rows),
    }
    # NeMo's tarred loader shuffles shard ORDER only; clips inside a shard are read in tar
    # order, and each batch comes from one shard. Shards must therefore be pre-mixed.
    worst = (0, None)
    for i, rs in enumerate(shard_rows):
        srcs = [r.get("source_dataset", "?") for r in rs]
        run, best = 1, 1
        for a, b in zip(srcs, srcs[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        if best > worst[0]:
            worst = (best, info["tars"][i].name)
    stats["longest_same_source_run"] = worst
    stats["kept_rows"] = kept
    return stats


def print_split_stats(info, stats):
    log(f"[data] {info['split']}: {info['n_shards']} shards -> {info['tar_pattern']}")
    m = info["manifest"]
    log(f"[data]   manifest_filepath = {m if isinstance(m, str) else f'[{len(m)} shard jsons]'}")
    log(f"[data]   clips {stats['n_total']} total, {stats['n_kept']} kept "
        f"({stats['hours_kept']:.1f} h); dropped {stats['n_too_short']} too short, "
        f"{stats['n_too_long']} > {stats['max_duration']:g} s (longest clip {stats['max_dur']:.1f} s)")
    sizes = stats["shard_sizes"]
    log(f"[data]   clips/shard min {min(sizes)} max {max(sizes)}")
    log("[data]   by source: " + ", ".join(f"{k}={v}" for k, v in sorted(stats["by_source"].items())))
    run, shard = stats["longest_same_source_run"]
    log(f"[data]   longest run of consecutive clips from one source: {run} (in {shard})")
    if info["split"] == "train" and run > 64:
        log("[data]   WARNING: NeMo's tarred loader only shuffles shard order, so long "
            "single-source runs inside a shard give single-source batches. Re-export with "
            "clips shuffled across shards if this is large.")
    if stats["not_nfkc"]:
        log(f"[data]   note: {stats['not_nfkc']} transcripts change under NFKC (the tokenizer "
            "normalizes NFKC internally; evaluate_nepali.py applies NFKC to both sides).")


# --------------------------------------------------------------------------------------
# Tokenizer checks
# --------------------------------------------------------------------------------------
def reference_pieces():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(REFERENCE_TOKENIZER))
    return [sp.id_to_piece(i) for i in range(sp.get_piece_size())]


def check_tokenizer_matches_reference(nemo_tokenizer):
    """Assert the model's SentencePiece vocabulary == assets/sravaani/tokenizer.model."""
    ref = reference_pieces()
    sp = getattr(nemo_tokenizer, "tokenizer", None)
    if sp is not None and hasattr(sp, "id_to_piece"):
        got = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    else:  # fall back to the generic NeMo API
        got = nemo_tokenizer.ids_to_tokens(list(range(nemo_tokenizer.vocab_size)))
    if got != ref:
        diff = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
        raise SystemExit(f"[tokenizer] model vocabulary ({len(got)} pieces) differs from "
                         f"{REFERENCE_TOKENIZER} ({len(ref)} pieces) at id {diff}. Refusing to train.")
    log(f"[tokenizer] model tokenizer == {REFERENCE_TOKENIZER.name} ({len(ref)} pieces) - OK")


def check_texts_encode_without_unk(nemo_tokenizer, rows_by_split):
    sp = getattr(nemo_tokenizer, "tokenizer", None)
    unk = sp.unk_id() if sp is not None and hasattr(sp, "unk_id") else 0
    bad, n, n_tok, n_words = [], 0, 0, 0
    for split, rows in rows_by_split.items():
        for r in rows:
            ids = nemo_tokenizer.text_to_ids(r["text"])
            n += 1
            n_tok += len(ids)
            n_words += len(r["text"].split())
            if unk in ids:
                bad.append((split, r["audio_filepath"], r["text"]))
    if bad:
        chars = Counter()
        for _, _, text in bad:
            for ch in set(text):
                if unk in nemo_tokenizer.text_to_ids(ch) and not ch.isspace():
                    chars[ch] += 1
        log(f"[tokenizer] {len(bad)}/{n} transcripts contain <unk>. Offending characters: "
            + ", ".join(f"{c!r} (U+{ord(c):04X}) x{k}" for c, k in chars.most_common(20)))
        for split, fid, text in bad[:5]:
            log(f"    {split} {fid}: {text[:100]}")
        raise SystemExit("[tokenizer] fix the transcripts (the SraVaani tokenizer has no digits "
                         "or most punctuation) before training")
    log(f"[tokenizer] {n} transcripts encode without <unk> "
        f"({n_tok / max(n_words, 1):.2f} tokens/word) - OK")


# --------------------------------------------------------------------------------------
# NeMo dataset config (shared by the real run and the dry run)
# --------------------------------------------------------------------------------------
# Representative train_ds of a NeMo FastConformer hybrid TDT-CTC BPE config. Only used by
# --dry-run without a checkpoint; a real run edits the restored model's own cfg.train_ds.
DRY_RUN_DS_TEMPLATE = {
    "manifest_filepath": None, "sample_rate": SAMPLE_RATE, "batch_size": 16,
    "shuffle": True, "num_workers": 8, "pin_memory": True, "use_start_end_token": False,
    "trim_silence": False, "max_duration": 20.0, "min_duration": 0.1, "is_tarred": False,
    "tarred_audio_filepaths": None, "shuffle_n": 2048, "bucketing_strategy": "synced_randomized",
    "bucketing_batch_size": None,
}

# Keys left over from pretraining that would switch NeMo to a different dataset class.
_DS_FORCE = {"use_lhotse": False, "is_concat": False, "use_dali": False,
             "is_code_switched": False, "use_semi_sorted_batching": False,
             "shard_manifests": False, "bucketing_batch_size": None, "bucketing_weights": None}
_DS_DROP = ("hf_data_cfg",)


def _augmentor_paths_missing(aug) -> list:
    from omegaconf import OmegaConf
    missing = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and ("path" in k or "manifest" in k):
                    if not all(Path(s).exists() for s in v.split(",")):
                        missing.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(OmegaConf.to_container(aug, resolve=True) if not isinstance(aug, (dict, list)) else aug)
    return missing


def configure_dataset(ds, info: dict, args, train: bool):
    """Point a NeMo *_ds config at our tarred shards (same keys as the official script)."""
    from omegaconf import ListConfig
    for k in _DS_DROP:
        ds.pop(k, None)
    for k, v in _DS_FORCE.items():
        if k in ds or v:
            ds[k] = v
    if train and ("augmentor" in ds or getattr(args, "augment_config", None)):
        import noise_augment  # registers the babble / music / phone perturbations
    if "augmentor" in ds and ds.augmentor is not None:
        missing = _augmentor_paths_missing(ds.augmentor)
        if not train or missing:
            log(f"[config] dropping {'validation' if not train else 'train'} augmentor"
                + (f" (files not found: {missing[:2]})" if missing else ""))
            ds.pop("augmentor")
        else:
            log(f"[config] keeping checkpoint's train augmentor: {dict(ds.augmentor)}")
    if train and getattr(args, "augment_config", None):
        ds.augmentor = noise_augment.load_augmentor(args.augment_config)
        log(f"[config] train augmentor from {args.augment_config}: {', '.join(ds.augmentor)}")
    m = info["manifest"]
    ds.manifest_filepath = m if isinstance(m, str) else ListConfig(list(m))
    ds.tarred_audio_filepaths = info["tar_pattern"]
    ds.is_tarred = True
    ds.shuffle = train
    if train:
        ds.shuffle_n = 2048  # WebDataset buffer (in this NeMo version it shuffles shard order)
    ds.batch_size = args.batch_size
    # Each worker reads whole shards, so more workers than shards would idle.
    ds.num_workers = max(1, min(args.num_workers, info["n_shards"]))
    ds.pin_memory = True
    # The checkpoint's validation_ds.max_duration is 20, which would silently drop clips.
    ds.max_duration = args.max_duration if train else args.val_max_duration
    ds.min_duration = args.min_duration
    if ds.get("sample_rate") is None:
        ds.sample_rate = SAMPLE_RATE
    if int(ds.sample_rate) != SAMPLE_RATE:
        raise SystemExit(f"[config] dataset sample_rate {ds.sample_rate} != {SAMPLE_RATE}")
    if "use_start_end_token" not in ds:
        ds.use_start_end_token = False
    return ds


def plan_schedule(n_train: int, args) -> dict:
    """max_steps exactly like the official script, from the (duration-filtered) manifest."""
    batches_per_epoch = math.ceil(n_train / args.batch_size)
    # Lightning also steps on the last, partial accumulation window -> ceil.
    steps_per_epoch = math.ceil(batches_per_epoch / args.accumulate)
    total = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    warmup = args.warmup_steps if args.warmup_steps > 0 else int(round(args.warmup_ratio * total))
    warmup = min(warmup, max(total - 1, 0))
    if isinstance(args.val_check_interval, float):
        vci = int(round(args.val_check_interval * batches_per_epoch))
    else:
        vci = args.val_check_interval
    # Align validation with optimizer-step boundaries and keep it inside one epoch.
    vci = max(args.accumulate, (vci // args.accumulate) * args.accumulate)
    vci = min(vci, batches_per_epoch)
    return {"batches_per_epoch": batches_per_epoch, "steps_per_epoch": steps_per_epoch,
            "max_steps": total, "warmup_steps": warmup, "val_check_batches": vci}


def print_schedule(plan, args):
    eff = args.batch_size * args.accumulate
    enc = "frozen" if args.freeze_encoder else f"lr {args.lr * args.encoder_lr_scale:.2e}"
    log(f"[plan] effective batch {eff} ({args.batch_size} x {args.accumulate}); "
        f"{plan['batches_per_epoch']} batches = {plan['steps_per_epoch']} optimizer steps/epoch")
    log(f"[plan] {args.epochs} epochs -> max_steps {plan['max_steps']}, warmup "
        f"{plan['warmup_steps']}, cosine {args.lr:.2e} -> {args.min_lr:.0e} (heads), encoder {enc}")
    log(f"[plan] validate every {plan['val_check_batches']} batches "
        f"(~{plan['val_check_batches'] / plan['batches_per_epoch']:.2f} epoch); precision {args.precision}")


# --------------------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------------------
def _pull_batches(dl, collection, tokenizer, n_batches, name):
    """Pull a few batches; with sample ids, check tokens and audio against the manifest."""
    import time
    t0 = time.time()
    checked = 0
    # Speed / time-stretch augmentation changes clip length on purpose; allow for it.
    lo = hi = 1.0
    aug = getattr(getattr(dl.dataset, "featurizer", None), "augmentor", None)
    for _, pert in getattr(aug, "_pipeline", []):
        if hasattr(pert, "_min_rate"):
            lo = min(lo, pert._min_rate, 1 / pert._max_rate)
            hi = max(hi, pert._max_rate, 1 / pert._min_rate)
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        signal, sig_len, tokens, tok_len = batch[:4]
        sample_ids = batch[4] if len(batch) > 4 else None
        secs = (sig_len.float() / SAMPLE_RATE).tolist()
        log(f"[dry-run] {name} batch {bi}: audio {tuple(signal.shape)} {signal.dtype} "
            f"({min(secs):.2f}-{max(secs):.2f} s), tokens {tuple(tokens.shape)} "
            f"(max len {int(tok_len.max())})")
        if not bool(signal.abs().max() > 0):
            raise SystemExit(f"[dry-run] {name}: batch {bi} audio is all zeros")
        if sample_ids is None:
            continue
        for j in range(signal.shape[0]):
            entry = collection[int(sample_ids[j])]
            ids = tokens[j, : int(tok_len[j])].tolist()
            if ids != tokenizer.text_to_ids(entry.text_raw):
                raise SystemExit(f"[dry-run] {name}: tokens of {entry.audio_file} do not match "
                                 "its manifest text")
            dur = float(sig_len[j]) / SAMPLE_RATE
            if entry.duration and not (entry.duration * lo - 0.05 <= dur <= entry.duration * hi + 0.05):
                raise SystemExit(f"[dry-run] {name}: {entry.audio_file} audio is {dur:.3f} s but "
                                 f"the manifest says {entry.duration:.3f} s")
            if j == 0:
                log(f"[dry-run]   {Path(str(entry.audio_file)).name}: "
                    f"{tokenizer.ids_to_text(ids)[:110]}")
            checked += 1
    log(f"[dry-run] {name}: OK ({time.time() - t0:.1f} s incl. worker start-up"
        + (f"; {checked} samples matched their manifest text and duration" if checked else "")
        + ")")


def dry_run(args, train_info, val_info, train_stats, val_stats, plan):
    import torch
    from omegaconf import DictConfig, OmegaConf, open_dict

    log("[dry-run] CUDA available: %s%s" % (
        torch.cuda.is_available(),
        f" ({torch.cuda.get_device_name(0)}, bf16 {torch.cuda.is_bf16_supported()})"
        if torch.cuda.is_available() else ""))

    if args.nemo_checkpoint is not None:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        log(f"[dry-run] restoring {args.nemo_checkpoint} on CPU ...")
        model = EncDecHybridRNNTCTCBPEModel.restore_from(str(args.nemo_checkpoint),
                                                         map_location=torch.device("cpu"))
        check_tokenizer_matches_reference(model.tokenizer)
        check_texts_encode_without_unk(model.tokenizer, {"train": train_stats["kept_rows"],
                                                         "validation": val_stats["kept_rows"]})
        tokenizer = model.tokenizer
        train_cfg = copy.deepcopy(model.cfg.train_ds)
        val_cfg = copy.deepcopy(model.cfg.validation_ds)
        owner = model
    else:
        from types import SimpleNamespace
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
        # Same class the model builds from its tokenizer.dir for tokenizer.type == bpe.
        tokenizer = SentencePieceTokenizer(model_path=str(REFERENCE_TOKENIZER))
        check_tokenizer_matches_reference(tokenizer)
        check_texts_encode_without_unk(tokenizer, {"train": train_stats["kept_rows"],
                                                   "validation": val_stats["kept_rows"]})
        train_cfg = DictConfig(copy.deepcopy(DRY_RUN_DS_TEMPLATE))
        val_cfg = DictConfig({**DRY_RUN_DS_TEMPLATE, "shuffle": False})
        owner = SimpleNamespace(tokenizer=tokenizer, local_rank=0, global_rank=0, world_size=1,
                                cfg=DictConfig({}))

    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    for name, cfg, info in (("train", train_cfg, train_info), ("validation", val_cfg, val_info)):
        with open_dict(cfg):
            configure_dataset(cfg, info, args, train=(name == "train"))
            cfg.return_sample_id = True  # dry-run only: lets us look up manifest entries
        log(f"[dry-run] {name}_ds:\n" + OmegaConf.to_yaml(cfg).rstrip())
        # The exact dataloader builder EncDecHybridRNNTCTCBPEModel.setup_training_data uses.
        dl = EncDecHybridRNNTCTCBPEModel._setup_dataloader_from_config(owner, cfg)
        ds = dl.dataset
        expected = (train_stats if name == "train" else val_stats)["n_kept"]
        log(f"[dry-run] {name}: {type(ds).__name__}, {len(ds)} clips after NeMo's duration "
            f"filter (expected {expected}), {cfg.num_workers} worker(s)")
        if len(ds) != expected:
            raise SystemExit(f"[dry-run] {name}: NeMo kept {len(ds)} clips, expected {expected}")
        _pull_batches(dl, ds.manifest_processor.collection, tokenizer, args.dry_run_batches, name)
    print_schedule(plan, args)
    check_free_disk(args.exp_dir, args)
    log("[dry-run] PASSED - data path is ready for training.")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def check_free_disk(path: Path, args):
    import shutil
    probe = path
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    need_gb = 5.4 * (args.save_top_k + 2) + 2 * 1.8  # top-k + last + one being written, 2 .nemo
    msg = (f"[disk] {free_gb:.1f} GB free at {probe}; this run needs up to ~{need_gb:.0f} GB "
           f"(each .ckpt ~5.4 GB, each .nemo ~1.8 GB)")
    if free_gb < 15 or free_gb < need_gb:
        log(msg + " - WARNING: low disk space, checkpoint saving may fail. Free space or "
            "lower --save-top-k.")
    else:
        log(msg)


def _prepare_exp_dir(args):
    record = args.exp_dir / "run_args.json"
    ckpt_dir = args.exp_dir / "checkpoints"
    has_ckpts = ckpt_dir.is_dir() and any(ckpt_dir.glob("*.ckpt"))
    current = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if args.resume:
        if not has_ckpts:
            raise SystemExit(f"[resume] no checkpoints in {ckpt_dir}")
        if record.is_file():
            old = json.loads(record.read_text())
            bad = [k for k in RESUME_STRUCTURAL_ARGS if old.get(k) != current.get(k)]
            if bad:
                raise SystemExit("[resume] these flags differ from the original run and change "
                                 "the optimizer layout: " + ", ".join(
                                     f"--{k.replace('_', '-')} {old.get(k)} (was) vs {current.get(k)}"
                                     for k in bad))
            for k in RESUME_SCHEDULE_ARGS:
                if old.get(k) != current.get(k):
                    log(f"[resume] WARNING: --{k.replace('_', '-')} was {old.get(k)}, now "
                        f"{current.get(k)} (the restored scheduler keeps its saved state)")
    elif has_ckpts:
        raise SystemExit(f"[exp] {ckpt_dir} already has checkpoints. Pass --resume to continue "
                         "that run, or choose a new --exp-dir.")
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        record.write_text(json.dumps(current, indent=2, ensure_ascii=False))


def _wandb_exp_manager_cfg(args):
    """exp_manager keys for an optional W&B logger (offline when no API key is available)."""
    if not args.wandb:
        return {}
    mode = args.wandb_mode or ("online" if os.environ.get("WANDB_API_KEY") else "offline")
    os.environ["WANDB_MODE"] = mode
    print(f"[wandb] project={args.wandb_project} name={args.wandb_name or args.run_name} mode={mode}", flush=True)
    return {
        "create_wandb_logger": True,
        "wandb_logger_kwargs": {
            "project": args.wandb_project,
            "name": args.wandb_name or args.run_name,
            "save_dir": str(args.exp_dir),
            "offline": mode == "offline",
            "tags": [t for t in args.wandb_tags.split(",") if t],
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        },
    }


def train(args, train_info, val_info, train_stats, val_stats, plan):
    import torch
    import lightning.pytorch as pl
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.utils.exp_manager import exp_manager
    from omegaconf import OmegaConf, open_dict

    _prepare_exp_dir(args)
    check_free_disk(args.exp_dir, args)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(args.nemo_checkpoint),
                                                     map_location=torch.device("cpu"))

    # ---- pre-flight: tokenizer must be the SraVaani one, transcripts must be covered ----
    check_tokenizer_matches_reference(model.tokenizer)
    check_texts_encode_without_unk(model.tokenizer, {"train": train_stats["kept_rows"],
                                                     "validation": val_stats["kept_rows"]})
    del train_stats["kept_rows"], val_stats["kept_rows"]

    if args.freeze_encoder:
        model.encoder.freeze()
        log("Encoder frozen - only decoder/joint/CTC weights will be updated.")
    else:
        log(f"Encoder trainable (lr x {args.encoder_lr_scale}).")

    if args.fused_batch_size > 0:
        # Same switches NeMo applies at init when cfg.joint.fuse_loss_wer is true.
        model.joint._fuse_loss_wer = True
        model.joint._fused_batch_size = args.fused_batch_size
        model.joint.set_loss(model.loss)
        model.joint.set_wer(model.wer)
        with open_dict(model.cfg):
            model.cfg.joint.fuse_loss_wer = True
            model.cfg.joint.fused_batch_size = args.fused_batch_size
    log(f"[model] joint fuse_loss_wer={model.joint.fuse_loss_wer} "
        f"fused_batch_size={getattr(model.joint, '_fused_batch_size', None)}; "
        f"ctc_loss_weight={model.cfg.get('aux_ctc', {}).get('ctc_loss_weight', '?')}")

    class MetricsLogger(pl.Callback):
        """Prints train loss every --log-every steps and val WER at every eval step
        (plus seconds per optimizer step and peak GPU memory, to tune batch size)."""

        def __init__(self, every):
            self.every = max(1, every)
            self._last = -1
            self._t = None

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            import time
            step = trainer.global_step
            if step == self._last or step % self.every:
                return
            now = time.time()
            rate = ""
            if self._t is not None and step > self._last >= 0:
                rate = f" {(now - self._t) / (step - self._last):.2f}s/step"
            self._last, self._t = step, now
            loss = trainer.callback_metrics.get("train_loss")
            if loss is not None:
                lrs = [g["lr"] for g in trainer.optimizers[0].param_groups]
                mem = torch.cuda.max_memory_reserved() / 2**30
                print(f"[step {step}] train_loss={loss.item():.4f} lr={max(lrs):.2e}{rate} "
                      f"peak_mem={mem:.1f}GiB", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            wer = trainer.callback_metrics.get("val_wer")
            loss = trainer.callback_metrics.get("val_loss")
            ctc = trainer.callback_metrics.get("val_wer_ctc")
            if wer is not None:
                msg = f"[step {trainer.global_step}] val_wer={wer.item():.4f}"
                if ctc is not None:
                    msg += f" val_wer_ctc={ctc.item():.4f}"
                if loss is not None:
                    msg += f" val_loss={loss.item():.4f}"
                print(msg, flush=True)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=args.precision,
        max_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        accumulate_grad_batches=args.accumulate,
        gradient_clip_val=1.0,
        val_check_interval=plan["val_check_batches"],
        log_every_n_steps=10,
        logger=False,               # exp_manager adds TensorBoard
        enable_checkpointing=False,  # exp_manager adds NeMo's ModelCheckpoint
        callbacks=[MetricsLogger(args.log_every)],
    )

    exp_manager(trainer, {
        "explicit_log_dir": str(args.exp_dir),
        "name": args.run_name,
        "create_tensorboard_logger": True,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": args.save_top_k,
            "save_last": True,
            "always_save_nemo": False,
            "save_nemo_on_train_end": False,  # we export best + last ourselves below
        },
        "resume_if_exists": args.resume,
        "resume_ignore_no_checkpoint": False,
        **_wandb_exp_manager_cfg(args),
    })
    # Attach the trainer before building dataloaders so NeMo can set the right number of
    # batches per epoch for the iterable tarred dataset (the official script skips this).
    model.set_trainer(trainer)

    with open_dict(model.cfg):
        # Training / validation data - tarred (official keys, paths discovered from disk)
        configure_dataset(model.cfg.train_ds, train_info, args, train=True)
        configure_dataset(model.cfg.validation_ds, val_info, args, train=False)

        # Optimizer - AdamW + CosineAnnealing as in the official script
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = args.lr
        model.cfg.optim.weight_decay = args.weight_decay
        model.cfg.optim.sched = OmegaConf.create({
            "name": "CosineAnnealing",
            "warmup_steps": plan["warmup_steps"],
            "max_steps": plan["max_steps"],
            "min_lr": args.min_lr,
        })
        model.cfg.pop("optim_param_groups", None)
        if not args.freeze_encoder and args.encoder_lr_scale != 1.0:
            # NeMo ModelPT turns this into a separate AdamW param group for model.encoder.
            model.cfg.optim_param_groups = {"encoder": {"lr": args.lr * args.encoder_lr_scale}}

    log("[config] train_ds:\n" + OmegaConf.to_yaml(model.cfg.train_ds).rstrip())
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_optimization(model.cfg.optim)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"[model] trainable params {n_train / 1e6:.1f}M / {n_all / 1e6:.1f}M")
    print_schedule(plan, args)

    trainer.fit(model)
    log(f"[train] done at step {trainer.global_step}; peak GPU memory reserved "
        f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB "
        f"(allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")

    # ---- export: last weights, then the best val_wer checkpoint ----
    last_path = args.exp_dir / f"{args.run_name}-last.nemo"
    model.save_to(str(last_path))
    log(f"[export] last weights -> {last_path}")
    ckpt_cb = trainer.checkpoint_callback
    best = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb is not None else ""
    if best and Path(best).is_file():
        state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=True)
        best_path = args.exp_dir / f"{args.run_name}-best.nemo"
        model.save_to(str(best_path))
        score = getattr(ckpt_cb, "best_model_score", None)
        log(f"[export] best checkpoint {Path(best).name} (val_wer "
            f"{float(score) if score is not None else float('nan'):.4f}) -> {best_path}")
    else:
        log("[export] no best checkpoint recorded; only the last weights were exported")


# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    # Variable-length batches fragment PyTorch's caching allocator (smoke test on the real
    # model: 30.7 GiB reserved for 19.7 GiB allocated). Expandable segments avoid most of
    # that waste. Must be set before CUDA is initialised.
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    numpy_compat()
    log(f"[data] data dir {args.data_dir} (manifest mode: {args.manifest_mode})")
    train_info = discover_split(args.data_dir, "train", args.manifest_mode)
    val_info = discover_split(args.data_dir, "validation", args.manifest_mode)
    train_stats = validate_split(train_info, args, args.max_duration)
    val_stats = validate_split(val_info, args, args.val_max_duration)
    print_split_stats(train_info, train_stats)
    print_split_stats(val_info, val_stats)
    if train_stats["n_kept"] == 0 or val_stats["n_kept"] == 0:
        raise SystemExit("[data] no clips left after duration filtering")
    plan = plan_schedule(train_stats["n_kept"], args)
    random.seed(args.seed)

    if args.dry_run:
        dry_run(args, train_info, val_info, train_stats, val_stats, plan)
    else:
        train(args, train_info, val_info, train_stats, val_stats, plan)


if __name__ == "__main__":
    sys.exit(main())
