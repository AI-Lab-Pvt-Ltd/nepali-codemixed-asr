#!/usr/bin/env python3
"""Export explicitly reviewed source paths; never copy local data or weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = (
    'README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md',
    'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md', 'SECURITY.md', '.gitignore',
    'requirements.txt', 'requirements-text.txt',
    'docs/EXPERIMENT_NOTES.md', 'docs/RELEASING.md',
    '.github/workflows/ci.yml', '.github/ISSUE_TEMPLATE/bug_report.md',
    '.github/pull_request_template.md',
    'scripts/finetune_nepali.py', 'scripts/evaluate_nepali.py',
    'scripts/sravaani_text.py', 'scripts/test_sravaani_text.py',
    'scripts/noise_augment.py', 'scripts/prepare_release.py',
)


def export(destination: Path) -> None:
    destination = destination.absolute()
    # Refuse any existing entry (including dangling symlinks); never overwrite.
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f'Destination already exists: {destination}')
    if destination.resolve().is_relative_to(ROOT):
        raise SystemExit('Choose a destination outside the source workspace.')
    for name in FILES:
        source = ROOT / name
        if not source.is_file() or any(p.is_symlink() for p in (source, *source.parents)):
            raise SystemExit(f'Missing, non-file or symlink source: {name}')
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    try:
        for name in FILES:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        (destination / 'RELEASE_MANIFEST.json').write_text(
            json.dumps({'sha256': hashes}, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        shutil.rmtree(destination)
        raise
    print(f'Exported {len(FILES)} source files to {destination}')
    print('Review docs/RELEASING.md before publication. Export is not a license or security audit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    export(parser.parse_args().destination)
