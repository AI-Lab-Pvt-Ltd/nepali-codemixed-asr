#!/usr/bin/env python
"""Evaluate a SraVaani .nemo (base or fine-tuned) on a tarred split (default: test).

Reads the audio straight from data/<split>/shard_NNNN.tar into memory (16 kHz mono WAV),
transcribes with NeMo's model.transcribe() (numpy arrays, supported by NeMo 2.7), decodes
with TDT (default) and/or the auxiliary CTC head, and reports WER / CER overall and per
source_dataset (jiwer, corpus-level), plus EN-WER: the WER over Latin-script (English)
words only, i.e. how well the code-mixed English is recognised.

Both sides get the same normalization: Unicode NFKC (the SraVaani SentencePiece model
applies NFKC internally, so hypotheses are NFKC) and whitespace collapsing. The data is
already lowercase, punctuation-free and digit-free.

Outputs (in --out-dir, default reports/):
  <tag>_<split>_<decoder>.jsonl   one line per clip: id, source_dataset, duration, ref, hyp
  <tag>_<split>.md                markdown table (overall + per source, all decoders)
  <tag>_<split>_metrics.json      the same numbers as JSON

Examples:
  python scripts/evaluate_nepali.py --nemo-checkpoint SraVaani-nemo-checkpoint.nemo --tag base
  python scripts/evaluate_nepali.py --nemo-checkpoint experiments/sravaani-nepali-ft/sravaani-nepali-best.nemo \
      --tag finetuned --decoder both
"""

import argparse
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_nepali import DEFAULT_DATA_DIR, PROJECT_DIR, SAMPLE_RATE, discover_split, numpy_compat, read_jsonl  # noqa: E402

LATIN_WORD = re.compile(r"[a-z]")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a SraVaani .nemo on a tarred split.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--nemo-checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="test")
    p.add_argument("--decoder", choices=("tdt", "ctc", "both"), default="tdt",
                   help="tdt = the model's main TDT (RNNT) decoder, ctc = auxiliary CTC head.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluate only the first N clips (in shard order) - quick checks.")
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "reports")
    p.add_argument("--tag", default=None, help="Output name prefix (default: checkpoint file stem).")
    p.add_argument("--input-mode", choices=("arrays", "files"), default="arrays",
                   help="arrays: pass numpy audio to transcribe(); files: extract each shard "
                        "to a temp dir and pass paths (fallback for older NeMo).")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (faster, tiny accuracy risk).")
    p.add_argument("--no-nfkc", action="store_true", help="Only collapse whitespace.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if not args.nemo_checkpoint.is_file():
        p.error(f"checkpoint not found: {args.nemo_checkpoint}")
    args.tag = args.tag or args.nemo_checkpoint.stem
    return args


def make_normalizer(nfkc: bool):
    def norm(text: str) -> str:
        if nfkc:
            text = unicodedata.normalize("NFKC", text)
        return " ".join(text.split())
    return norm


def load_entries(info, limit):
    """Manifest rows grouped by shard, in tar order (the data contract)."""
    per_shard, total = [], 0
    for tar_path, json_path in zip(info["tars"], info["shard_jsons"]):
        rows = read_jsonl(json_path)
        if limit:
            rows = rows[: max(0, limit - total)]
        total += len(rows)
        if rows:
            per_shard.append((tar_path, rows))
        if limit and total >= limit:
            break
    return per_shard, total


def read_shard_audio(tar_path, wanted):
    """{member name: float32 mono numpy array} for the wanted members of one shard."""
    import numpy as np
    import soundfile as sf
    out = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.name not in wanted:
                continue
            audio, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                raise SystemExit(f"{tar_path.name}/{m.name}: {sr} Hz, expected {SAMPLE_RATE}")
            out[m.name] = np.ascontiguousarray(audio)
            if len(out) == len(wanted):
                break
    missing = wanted - out.keys()
    if missing:
        raise SystemExit(f"{tar_path.name}: {len(missing)} manifest entries not in the tar, "
                         f"e.g. {sorted(missing)[:3]}")
    return out


def hyp_texts(result):
    """transcribe() returns list[Hypothesis] (NeMo 2.x) or (best, all) / list[str] (older)."""
    if isinstance(result, tuple):
        result = result[0]
    return [h.text if hasattr(h, "text") else str(h) for h in result]


def transcribe_shard(model, audio_by_name, names, args):
    import soundfile as sf
    import torch
    # Longest first: similar lengths share a batch (less padding), and an OOM shows up early.
    order = sorted(names, key=lambda n: -len(audio_by_name[n]))
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else torch.autocast("cuda", enabled=False)
    with ctx:
        if args.input_mode == "arrays":
            hyps = hyp_texts(model.transcribe([audio_by_name[n] for n in order],
                                              batch_size=args.batch_size, verbose=False))
        else:
            with tempfile.TemporaryDirectory(prefix="sravaani_eval_") as tmp:
                paths = []
                for n in order:
                    path = os.path.join(tmp, n)
                    sf.write(path, audio_by_name[n], SAMPLE_RATE, subtype="PCM_16")
                    paths.append(path)
                hyps = hyp_texts(model.transcribe(paths, batch_size=args.batch_size, verbose=False))
    if len(hyps) != len(order):
        raise SystemExit(f"transcribe returned {len(hyps)} results for {len(order)} inputs")
    return dict(zip(order, hyps))


def score(pairs):
    """Corpus-level WER/CER and EN-WER (Latin-script words only) for [(ref, hyp, dur)]."""
    import jiwer
    pairs = [p for p in pairs if p[0]]
    if not pairs:
        return None
    refs = [p[0] for p in pairs]
    hyps = [p[1] for p in pairs]
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    en_refs, en_hyps = [], []
    for r, h in zip(refs, hyps):
        er = " ".join(t for t in r.split() if LATIN_WORD.search(t))
        if er:
            en_refs.append(er)
            en_hyps.append(" ".join(t for t in h.split() if LATIN_WORD.search(t)))
    en = jiwer.process_words(en_refs, en_hyps) if en_refs else None
    n_words = w.hits + w.substitutions + w.deletions
    return {
        "n": len(pairs), "hours": sum(p[2] for p in pairs) / 3600.0, "ref_words": n_words,
        "wer": w.wer, "cer": c.cer, "sub": w.substitutions, "del": w.deletions, "ins": w.insertions,
        "en_wer": en.wer if en else None,
        "en_ref_words": (en.hits + en.substitutions + en.deletions) if en else 0,
    }


def fmt_pct(x):
    return "-" if x is None else f"{100 * x:.2f}"


def write_markdown(path, args, metrics, decoders, n_clips, n_empty):
    lines = [
        f"# {args.tag} on {args.split}",
        "",
        f"- checkpoint: `{args.nemo_checkpoint}`",
        f"- data: `{args.data_dir / args.split}` ({n_clips} clips"
        + (f", first {args.limit} only" if args.limit else "") + ")",
        f"- normalization: {'NFKC + ' if not args.no_nfkc else ''}whitespace collapse on ref and hyp"
        + (f"; {n_empty} clips with empty reference skipped" if n_empty else ""),
        "- WER/CER are corpus-level (errors / reference words or characters, spaces count as "
        "characters); EN-WER = WER over Latin-script words only (clips whose reference has any).",
        "",
    ]
    head = "| subset | clips | hours |" + "".join(
        f" {d.upper()} WER % | {d.upper()} CER % | {d.upper()} EN-WER % |" for d in decoders)
    lines += [head, "|" + "---|" * (3 + 3 * len(decoders))]
    subsets = ["ALL"] + sorted(k for k in metrics[decoders[0]] if k != "ALL")
    for sub in subsets:
        base = metrics[decoders[0]][sub]
        row = f"| {'**all**' if sub == 'ALL' else sub} | {base['n']} | {base['hours']:.2f} |"
        for d in decoders:
            m = metrics[d][sub]
            row += f" {fmt_pct(m['wer'])} | {fmt_pct(m['cer'])} | {fmt_pct(m['en_wer'])} |"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    numpy_compat()
    norm = make_normalizer(not args.no_nfkc)
    info = discover_split(args.data_dir.resolve(), args.split, "combined")
    per_shard, n_clips = load_entries(info, args.limit)
    print(f"[eval] {args.split}: {n_clips} clips from {len(per_shard)} shard(s) of {info['n_shards']}", flush=True)

    import torch
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict

    model = ASRModel.restore_from(str(args.nemo_checkpoint), map_location=torch.device("cpu"))
    model = model.to(args.device).eval()
    if "sample_rate" not in model.cfg:  # needed by transcribe() for numpy input
        with open_dict(model.cfg):
            model.cfg.sample_rate = SAMPLE_RATE
    decoders = ["tdt", "ctc"] if args.decoder == "both" else [args.decoder]
    if "ctc" in decoders and not hasattr(model, "ctc_decoder"):
        raise SystemExit("this model has no CTC head; use --decoder tdt")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics, n_empty = {}, 0
    for dec in decoders:
        if hasattr(model, "ctc_decoder"):  # hybrid model: choose the decoding head
            model.change_decoding_strategy(decoder_type="rnnt" if dec == "tdt" else "ctc", verbose=False)
        t0, audio_secs = time.time(), 0.0
        records = []
        for tar_path, rows in per_shard:
            audio = read_shard_audio(tar_path, {r["audio_filepath"] for r in rows})
            audio_secs += sum(len(a) for a in audio.values()) / SAMPLE_RATE
            hyps = transcribe_shard(model, audio, [r["audio_filepath"] for r in rows], args)
            for r in rows:
                name = r["audio_filepath"]
                records.append({"id": name.rsplit(".", 1)[0],
                                "source_dataset": r.get("source_dataset", "unknown"),
                                "duration": float(r["duration"]),
                                "ref": norm(r["text"]), "hyp": norm(hyps[name])})
            print(f"[eval] {dec}: {tar_path.name} done ({len(records)}/{n_clips})", flush=True)
        elapsed = time.time() - t0
        out_jsonl = args.out_dir / f"{args.tag}_{args.split}_{dec}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_empty = sum(1 for rec in records if not rec["ref"])
        groups = defaultdict(list)
        for rec in records:
            item = (rec["ref"], rec["hyp"], rec["duration"])
            groups["ALL"].append(item)
            groups[rec["source_dataset"]].append(item)
        metrics[dec] = {k: score(v) for k, v in groups.items()}
        metrics[dec] = {k: v for k, v in metrics[dec].items() if v is not None}
        m = metrics[dec]["ALL"]
        print(f"[eval] {dec}: WER {fmt_pct(m['wer'])}%  CER {fmt_pct(m['cer'])}%  "
              f"EN-WER {fmt_pct(m['en_wer'])}%  ({m['n']} clips, RTFx {audio_secs / elapsed:.0f}) "
              f"-> {out_jsonl}", flush=True)

    md = args.out_dir / f"{args.tag}_{args.split}.md"
    write_markdown(md, args, metrics, decoders, n_clips, n_empty)
    js = args.out_dir / f"{args.tag}_{args.split}_metrics.json"
    js.write_text(json.dumps({"checkpoint": str(args.nemo_checkpoint), "split": args.split,
                              "limit": args.limit, "nfkc": not args.no_nfkc,
                              "metrics": metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))
    print(f"[eval] wrote {md} and {js}")


if __name__ == "__main__":
    sys.exit(main())
