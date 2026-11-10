# SraVaani Nepali fine-tuning

An open source contribution from **AI Lab Pvt. Ltd.** for adapting SraVaani speech recognition to Nepali and Nepali–English code-mixed speech using NVIDIA NeMo.

The core workflow provides tarred-dataset checks, tokenizer compatibility checks, fine-tuning with optional encoder freezing, checkpoint resume, best-model export, and evaluation with WER, CER and English-word WER.

Company-authored code is licensed under [Apache-2.0](LICENSE). Model weights, datasets, tokenizers and third-party code retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). This release does not include training audio, transcripts or model weights.

## Setup

Use Linux, Python 3.12 and a CUDA-capable NVIDIA GPU for training. The original experiments used a 32 GB RTX 5090; smaller GPUs require smaller batches. GPU training and dependency installation have not been revalidated for this release.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for your GPU/CUDA environment first.
python -m pip install -r requirements.txt
```

The requirements describe the core SraVaani workflow. The local research workspace also contains Whisper, IndicConformer and dataset-preparation experiments; those are outside the curated release.

## Bring your model and data

Obtain the SraVaani checkpoint and matching SentencePiece tokenizer from the [SraVaani publisher](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0), following the publisher's access process and terms. Store the checkpoint locally and configure the tokenizer path:

```bash
export SRAVAANI_TOKENIZER=/absolute/path/to/tokenizer.model
```

Prepare `data/train`, `data/validation` and `data/test`. Each split contains uncompressed `shard_0000.tar`, `shard_0001.tar`, etc., corresponding `shard_0000.json` JSONL manifests, and `manifest.json` containing those manifests concatenated in shard order. Tar members must follow manifest order. Each WAV is 16 kHz, mono, PCM 16-bit. For example, a manifest line is:

```json
{"audio_filepath":"example.wav","duration":1.5,"text":"नमस्ते","source_dataset":"your_dataset","shard_id":0}
```

Use your own appropriately licensed data. Match the checkpoint's text convention: Nepali in Devanagari, English in lowercase Latin, numbers spelled out, no punctuation or digits. The training script verifies tokenizer vocabulary and transcript coverage. Keep speakers and recordings separated across splits to avoid evaluation leakage.

## Train and evaluate

```bash
python scripts/finetune_nepali.py --help
python scripts/finetune_nepali.py --nemo-checkpoint /path/to/base.nemo --data-dir data --dry-run
python scripts/finetune_nepali.py --nemo-checkpoint /path/to/base.nemo --data-dir data \
  --exp-dir experiments/nepali
python scripts/evaluate_nepali.py \
  --nemo-checkpoint experiments/nepali/sravaani-nepali-best.nemo \
  --data-dir data --decoder both --tag finetuned
```

The defaults train for 10 epochs with batch size 8, gradient accumulation 8 and bf16 precision. If GPU memory is insufficient, try `--batch-size 4 --accumulate 16`. Training exports best and last `.nemo` models plus Lightning checkpoints. Resume the same run with the same arguments and `--resume`. Evaluation writes aggregate metrics and per-clip predictions to `reports/`; predictions may contain sensitive speech content.

For text normalization only:

```bash
python -m pip install -r requirements-text.txt
python scripts/test_sravaani_text.py
```

Text tests require the matching tokenizer configured above. The text normalization module offers candidate number verbalizations; selecting the intended reading may require audio evidence. It is not a general inverse text normalization system.

## Contributing and release

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and our [Code of Conduct](CODE_OF_CONDUCT.md). For the research workspace's original measurements and recipes, see [experiment notes](docs/EXPERIMENT_NOTES.md); those notes describe local artifacts and historical runs, not files supplied with this release.

Maintainers: use the [release checklist](docs/RELEASING.md) and `python scripts/prepare_release.py /tmp/sravaani-public` to create a clean source distribution from this research workspace. Review that directory before publishing it as a new repository.

## Citation

If you find this repository or codebase helpful in your research, please cite:

```bibtex
@misc{ailab2026nepalicodemixedasr,
  author = {{AI Lab Pvt. Ltd.}},
  title = {Nepali and Nepali-English Code-Mixed Speech Recognition with SraVaani},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI-Lab-Pvt-Ltd/nepali-codemixed-asr}}
}
```
