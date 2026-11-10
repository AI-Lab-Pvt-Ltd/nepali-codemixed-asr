# Contributing

We welcome bug reports, documentation improvements, and reproducible improvements to Nepali/code-mixed ASR.

Open an issue describing the expected and actual behavior, command, Python/NeMo/PyTorch versions and GPU. Use synthetic examples and redact transcripts, paths and credentials. Discuss major changes in an issue before implementing them.

For a pull request, describe the problem, the change and how you verified it. Keep changes focused. Do not submit model weights, datasets, recordings, private transcripts, credentials or generated experiment outputs. Contributions must be yours to submit under Apache-2.0; retain applicable third-party notices and attribution.

From the repository root:

```bash
python -m compileall -q scripts
python scripts/finetune_nepali.py --help
python scripts/evaluate_nepali.py --help
```

For normalization changes, install `requirements-text.txt`, configure `SRAVAANI_TOKENIZER` as described in the README and run `python scripts/test_sravaani_text.py`. For training changes, report a dry run and a short training/evaluation run on data you may use, including environment details. CPU CI covers syntax and CLI entry points; it does not establish GPU correctness or model quality.
