# Nepali–English Code-Mixed Speech Recognition: Data Preparation and Adaptation of a Speech Foundation Model

[![Core checks](https://github.com/AI-Lab-Pvt-Ltd/nepali-codemixed-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/AI-Lab-Pvt-Ltd/nepali-codemixed-asr/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official code and data preparation pipeline for the paper:
> **Nepali–English Code-Mixed Speech Recognition: Data Preparation and Adaptation of a Speech Foundation Model**  
> *Jeevan Neupane, Janardan Bhetwal, Kshitiz Paudel, Anup Aryal, Bipin Bashyal*  
> **AI Lab Pvt. Ltd., Nepal**  
> Contact: `{jeevanneupane, janardanbhetwal, kshitizpaudel, anuparyal, bipinbashyal}@ailab.com.np`

---

## Abstract

Nepali speakers regularly mix English words into conversation, but multilingual speech recognisers often struggle with language switches and the writing script used for English words. We study how to adapt SraVaani, a 449M-parameter hybrid TDT–CTC model, using a 172.8-hour corpus covering six Nepali speech domains. Its pretrained tokenizer does not support digits and several common symbols. Rather than replace it or discard these recordings, we generate Nepali and English readings for numbers and use zero-shot ASR output to select a plausible form. Full fine-tuning reduces test word error rate (WER) from 48.6% to 17.1%; a 12-epoch run reaches 16.3%, compared with 16.4% for fine-tuned Whisper large-v3-turbo. Freezing the encoder increases WER to 25.5%, while excluding numeric clips raises WER on numeric test speech from 14.3% to 21.3%. We also evaluate on FLEURS using both its original references and mixed-script alternatives. The comparison shows that transcription conventions explain part, but not all, of the out-of-domain performance loss.

### Key Benchmark Results

| Model / Configuration | Test WER (%) | Note |
|---|---|---|
| SraVaani (Zero-shot base) | 48.6% | Base model without adaptation |
| SraVaani (Frozen Encoder, 2 epochs) | 25.5% | Official recipe with frozen encoder |
| SraVaani (Full Fine-tuning, 10 epochs) | 17.1% | End-to-end with encoder LR scale 0.5 |
| **SraVaani (Full Fine-tuning, 12 epochs)** | **16.3%** | **Best model** |
| Whisper large-v3-turbo (Fine-tuned) | 16.4% | Baseline comparison |

- **Number verbalization impact:** Excluding numeric clips degrades WER on numeric test speech from 14.3% to 21.3%.
- Company-authored code is licensed under [Apache-2.0](LICENSE). Model weights, datasets, tokenizers and third-party code retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). This release does not include training audio, transcripts or model weights.

---

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

For text normalization and number verbalization:

```bash
python -m pip install -r requirements-text.txt
python scripts/test_sravaani_text.py
```

Text tests require the matching tokenizer configured above. The text normalization module offers candidate number verbalizations; selecting the intended reading may require audio evidence. It is not a general inverse text normalization system.

## Contributing and release

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and our [Code of Conduct](CODE_OF_CONDUCT.md). For the research workspace's original measurements and recipes, see [experiment notes](docs/EXPERIMENT_NOTES.md); those notes describe local artifacts and historical runs, not files supplied with this release.

Maintainers: use the [release checklist](docs/RELEASING.md) and `python scripts/prepare_release.py /tmp/sravaani-public` to create a clean source distribution from this research workspace. Review that directory before publishing it as a new repository.

## Citation

If you use this work, codebase, or findings in your research, please cite our paper:

```bibtex
@article{neupane2026nepali,
  title={Nepali--English Code-Mixed Speech Recognition: Data Preparation and Adaptation of a Speech Foundation Model},
  author={Neupane, Jeevan and Bhetwal, Janardan and Paudel, Kshitiz and Aryal, Anup and Bashyal, Bipin},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/AI-Lab-Pvt-Ltd/nepali-codemixed-asr}
}
```
