# Public release checklist

The research workspace is not the public distribution. Export only the reviewed core source; the exporter deliberately excludes datasets, weights, upstream snapshots, papers and local experiment scripts. A `.gitignore` does not remove tracked files or history.

## Decisions and remaining blockers

- Company attribution: AI Lab Pvt. Ltd. License for company contributions: Apache-2.0, as selected by the company representative.
- Verify company ownership and contributor permissions for the exported source.
- **Resolve upstream recipe provenance before publication.** `finetune_nepali.py` derives from the official SraVaani recipe. Record its exact upstream source/revision, license or written permission, and required notices in THIRD_PARTY_NOTICES.md. Do not assume that access to weights or a public source file grants redistribution rights.
- Publish a monitored private security and conduct reporting channel; update SECURITY.md and CODE_OF_CONDUCT.md. Configure GitHub private vulnerability reporting if available.
- Install the documented requirements in a fresh Python 3.12 environment and validate against the supported GPU/CUDA stack. Requirements are starting constraints, not a tested lockfile. Record resolved versions and a successful dry run, short training, checkpoint resume and evaluation before claiming a supported environment.

## Prepare and inspect

```bash
python scripts/prepare_release.py /tmp/sravaani-public
cd /tmp/sravaani-public
python -m compileall -q scripts
python scripts/finetune_nepali.py --help
python scripts/evaluate_nepali.py --help
```

The destination must not exist. The exporter uses an explicit file list, rejects symlink sources and writes RELEASE_MANIFEST.json with SHA-256 hashes. It does not copy Git history. Inspect every exported file for private information and review the historical experiment notes for publication suitability. Remove or redact those notes from the exporter if needed, retaining appropriate attribution elsewhere.

Run your company's secret scanner on the export. Scan Git history separately if publishing an existing repository. Any exposed credentials must be revoked; deleting a file is insufficient. Review data and model permissions separately before making any future asset release.

The local historical notes describe experiments and dataset statistics, not a reproducible bundled dataset. Do not present them as independently validated benchmark claims for the public release.

## Publish after resolving blockers

Create a new company-owned repository from the reviewed export, configure maintainers and reporting channels, enable CI, and tag the initial release. If importing history instead, audit it first. Do not upload the entire research directory or initialize a public repository there.

The license applies to the source contribution; access to base models and matching tokenizers remains a separate publisher process. Publication is a separate action from preparing this export.
