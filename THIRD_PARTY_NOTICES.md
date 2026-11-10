# Third-party provenance

Apache-2.0 applies to AI Lab Pvt. Ltd.'s contributions. It does not grant rights to third-party models, data, copied source, or other assets.

- **SraVaani**, published by ARTPARK/IISc: https://huggingface.co/ARTPARK-IISc/SraVaani-1.0. `scripts/finetune_nepali.py` was developed from the official SraVaani fine-tuning recipe and uses the same NeMo training APIs. Maintainers must verify the original recipe's license, permission for derivative distribution, and any required notices before publishing. This provenance note is not permission to redistribute.
- **NVIDIA NeMo**: https://github.com/NVIDIA/NeMo. An external runtime dependency; its license and dependency notices remain applicable.
- The local `assets/sravaani-github/` and `external/` directories contain upstream material. They are excluded from the curated source release pending license and provenance review.
- Base/fine-tuned weights, tokenizer files and configs under `assets/sravaani/`, datasets, audio, transcripts, predictions, experiment logs, and paper assets are excluded. Obtain any needed assets separately under their respective terms. The code license does not establish redistribution rights for them.

Dependencies are installed separately; review their licenses for your distribution. Before removing a release blocker, record the upstream revision, applicable license, and required attribution here.
