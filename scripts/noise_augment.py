"""On-the-fly audio augmentation for the NeMo fine-tuning scripts (training clips only).

Importing this module adds three names to NeMo's perturbation registry, so a train_ds.augmentor
YAML (see your augmentor.yaml) can use them next to NeMo's own
`speed`, `impulse`, `noise`, `gain`, ...:
  * babble - crowd babble; NeMo's NoisePerturbation with its own manifest and SNR range,
  * music  - background music; same class,
  * phone  - phone / telemedicine call channel: 8 kHz, a random ~300-3400 Hz band, and
             optionally G.711-style 8-bit mu-law companding, then back to the input rate.
The transcript is never changed. Validation and test data are never augmented
(finetune_nepali.configure_dataset drops any validation augmentor).
"""
import random
from math import gcd
from pathlib import Path

import numpy as np
from nemo.collections.asr.parts.preprocessing.perturb import (
    NoisePerturbation, Perturbation, perturbation_types, register_perturbation)
from scipy.signal import butter, resample_poly, sosfilt

PHONE_SR = 8000


class PhonePerturbation(Perturbation):
    """Telephone-band channel; NeMo's TranscodePerturbation needs the sox CLI, this does not."""

    def __init__(self, low_hz=(200.0, 400.0), high_hz=(3000.0, 3600.0), mulaw_prob=0.7, rng=None):
        self._low = (float(low_hz[0]), float(low_hz[1]))
        self._high = (float(high_hz[0]), float(high_hz[1]))
        self._mulaw_prob = float(mulaw_prob)
        if not 0 < self._low[0] <= self._low[1] < self._high[0] <= self._high[1] < PHONE_SR / 2:
            raise ValueError(f"phone: bad band low_hz={low_hz} high_hz={high_hz}")
        random.seed(rng) if rng else None

    def perturb(self, data):
        x = data._samples
        if x.ndim != 1 or len(x) == 0 or not np.any(x):
            return
        g = gcd(int(data.sample_rate), PHONE_SR)
        up, down = PHONE_SR // g, int(data.sample_rate) // g
        y = resample_poly(x, up, down)
        band = [random.uniform(*self._low), random.uniform(*self._high)]
        y = sosfilt(butter(4, band, btype="bandpass", fs=PHONE_SR, output="sos"), y)
        if random.random() < self._mulaw_prob:
            mu, peak = 255.0, float(np.max(np.abs(y))) or 1.0
            c = np.sign(y) * np.log1p(mu * np.abs(y / peak)) / np.log1p(mu)
            c = np.round((c + 1) / 2 * mu) / mu * 2 - 1  # 8-bit code words
            y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu * peak
        y = resample_poly(y, down, up)[:len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        data._samples = y.astype(x.dtype)


for _name, _cls in (("babble", NoisePerturbation), ("music", NoisePerturbation),
                    ("phone", PhonePerturbation)):
    if _name not in perturbation_types:
        register_perturbation(_name, _cls)


def load_augmentor(path):
    """Read a train_ds.augmentor YAML and check every name and file it refers to."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(path))
    unknown = [k for k in cfg if k not in perturbation_types]
    missing = [p for k in cfg for key in ("manifest_path",) if key in cfg[k]
               for p in str(cfg[k][key]).split(",") if not Path(p).exists()]
    no_prob = [k for k in cfg if "prob" not in cfg[k]]
    if unknown or missing or no_prob:
        raise SystemExit(f"[augment] {path}: unknown {unknown}, missing files {missing}, "
                         f"no prob {no_prob}")
    return cfg
