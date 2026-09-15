"""Branch B feature extraction: magnitude spectrum → sound-design descriptors.

Pure numpy, deliberately. Everything Branch B needs — a windowed STFT, spectral
moments, peak picking, an HPS pitch estimate — is a few lines of ``np.fft``, so
the sound-design path runs on the **core install** rather than behind the
``audio`` extra. That matters because Branch B is the one stage whose inputs
(the harmonic stem) are already in memory as arrays: there is nothing left to
decode, so there is nothing to require.

Every function here is a pure array → number mapping and is unit-tested directly
against synthesized signals with known answers.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# STFT                                                                        #
# --------------------------------------------------------------------------- #
def stft_magnitude(
    y: np.ndarray, sr: int, n_fft: int = 8192, hop_length: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs, magnitude, times)`` for a Hann-windowed STFT.

    ``magnitude`` is ``(n_bins, n_frames)``. A signal shorter than one frame is
    zero-padded to a single frame, so every span yields at least one spectrum.

    ``n_fft`` defaults high (8192 ≈ 2.7 Hz bins at 22.05 kHz) because the
    detuning estimate needs to resolve unison voices only tens of cents apart;
    that costs frequency-vs-time resolution, which is the right trade for
    sustained synth material.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    window = np.hanning(n_fft)
    starts = range(0, y.size - n_fft + 1, hop_length)
    frames = np.stack([np.fft.rfft(y[s : s + n_fft] * window) for s in starts], axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.array([(s + n_fft / 2) / sr for s in starts])
    return freqs, np.abs(frames), times


def frame_energy(magnitude: np.ndarray) -> np.ndarray:
    """Per-frame energy, used to weight frame statistics away from silence."""
    return (magnitude**2).sum(axis=0)


# --------------------------------------------------------------------------- #
# Spectral moments                                                            #
# --------------------------------------------------------------------------- #
def spectral_centroid(freqs: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Per-frame centre of spectral mass, in Hz (perceptual "brightness")."""
    total = magnitude.sum(axis=0) + _EPS
    return (freqs[:, None] * magnitude).sum(axis=0) / total


def spectral_bandwidth(
    freqs: np.ndarray, magnitude: np.ndarray, centroid: np.ndarray | None = None
) -> np.ndarray:
    """Per-frame spread of spectral mass about the centroid, in Hz."""
    if centroid is None:
        centroid = spectral_centroid(freqs, magnitude)
    total = magnitude.sum(axis=0) + _EPS
    dev = (freqs[:, None] - centroid[None, :]) ** 2
    return np.sqrt((dev * magnitude).sum(axis=0) / total)


def spectral_flatness(magnitude: np.ndarray) -> np.ndarray:
    """Per-frame geometric/arithmetic mean of power: ~0 tonal, →1 noise-like."""
    power = magnitude**2 + _EPS
    geo = np.exp(np.log(power).mean(axis=0))
    arith = power.mean(axis=0)
    return geo / arith


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Energy-weighted mean, so near-silent frames cannot drag a statistic."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= _EPS:
        return float(np.mean(values))
    return float((np.asarray(values, dtype=float) * w).sum() / w.sum())


def trend_per_second(values: np.ndarray, times: np.ndarray, min_r: float = 0.5) -> float:
    """Least-squares slope of ``values`` over ``times`` (units/second).

    Applied to the centroid track this is the *filter movement* descriptor: a
    sweep opening reads positive, a closing filter negative, a static patch 0.0.

    The slope is screened for linearity and returned as 0.0 when
    ``|r| < min_r``. Without that screen, beating between detuned unison voices
    wobbles the centroid enough to fit a large but meaningless slope; a real
    sweep is a strongly linear trend, so this reports movement, not jitter.
    """
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(y) < 3 or float(np.ptp(x)) <= 0 or float(np.ptp(y)) <= 0:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope) if abs(r) >= min_r else 0.0


# --------------------------------------------------------------------------- #
# Partials                                                                    #
# --------------------------------------------------------------------------- #
def find_peaks(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    rel_threshold: float = 0.05,
    max_peaks: int = 64,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick spectral peaks with parabolic (log-magnitude) interpolation.

    Returns ``(frequencies_hz, magnitudes)`` sorted by frequency. Interpolation
    matters here: a 2.7 Hz bin is coarser than the detuning we are trying to
    measure, and the parabolic fit recovers most of the sub-bin position.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.size < 3 or spectrum.max() <= _EPS:
        return np.empty(0), np.empty(0)
    floor = rel_threshold * spectrum.max()
    interior = np.arange(1, spectrum.size - 1)
    is_peak = (
        (spectrum[interior] > spectrum[interior - 1])
        & (spectrum[interior] >= spectrum[interior + 1])
        & (spectrum[interior] >= floor)
    )
    idx = interior[is_peak]
    if fmax is not None:
        idx = idx[freqs[idx] <= fmax]
    if idx.size == 0:
        return np.empty(0), np.empty(0)
    if idx.size > max_peaks:  # keep the strongest, then restore frequency order
        idx = np.sort(idx[np.argsort(spectrum[idx])[-max_peaks:]])

    log = np.log(spectrum + _EPS)
    a, b, c = log[idx - 1], log[idx], log[idx + 1]
    denom = a - 2 * b + c
    shift = np.where(np.abs(denom) < _EPS, 0.0, 0.5 * (a - c) / np.where(denom == 0, _EPS, denom))
    shift = np.clip(shift, -0.5, 0.5)
    bin_hz = float(freqs[1] - freqs[0])
    return freqs[idx] + shift * bin_hz, spectrum[idx]


def estimate_f0(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    f0_min: float = 40.0,
    f0_max: float = 1200.0,
    n_harmonics: int = 5,
    subharmonic_tol: float = 0.05,
) -> float:
    """Harmonic-product-spectrum f0 estimate, in Hz (0.0 if none is found).

    HPS multiplies the spectrum by decimated copies of itself, so bins supported
    by a whole harmonic series survive and lone partials do not. On genuinely
    inharmonic material it returns whatever fits best — which is fine, because
    :func:`inharmonicity` then reports a large deviation against it.

    Two corrections keep it from octave-erring, which raw HPS does readily on
    any spectrum with strong even harmonics (every saw):

    * it runs on a 3-bin max envelope, not the raw spectrum. At this FFT size a
      partial rarely sits on an integer bin, and decimation then samples the
      skirt of the true peak; whether it does depends on the fractional bin
      offset, which *grows with the candidate f0* — a bias straight towards the
      octave above. The envelope removes it.
    * each of f0/2 and f0/3 is still checked afterwards, and the lowest
      subharmonic carrying ``subharmonic_tol`` of the winner's score wins.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.max() <= _EPS:
        return 0.0
    envelope = np.maximum(
        spectrum, np.maximum(np.roll(spectrum, 1), np.roll(spectrum, -1))
    )
    hps = envelope.copy()
    for k in range(2, n_harmonics + 1):
        decimated = envelope[::k]
        hps[: decimated.size] *= decimated
    band = (freqs >= f0_min) & (freqs <= f0_max)
    if not band.any():
        return 0.0
    masked = np.where(band, hps, 0.0)
    best = float(masked.max())
    if best <= _EPS:
        return 0.0
    f_best = float(freqs[int(np.argmax(masked))])

    for divisor in (3, 2):  # lowest plausible fundamental first
        cand = f_best / divisor
        if cand < f0_min:
            continue
        idx = int(np.argmin(np.abs(freqs - cand)))
        lo, hi = max(idx - 2, 0), min(idx + 3, hps.size)  # tolerate bin rounding
        if float(hps[lo:hi].max()) >= subharmonic_tol * best:
            return float(freqs[lo + int(np.argmax(hps[lo:hi]))])
    return f_best


def refine_f0(
    f0: float,
    peak_freqs: np.ndarray,
    peak_mags: np.ndarray,
    tol_cents: float = 50.0,
    min_rel_mag: float = 0.3,
    f_min: float = 40.0,
) -> float:
    """Snap an HPS estimate onto a partial actually present in the spectrum.

    The fundamental should coincide with a detected peak. When it does (within
    ``tol_cents``) the peak's interpolated frequency is the better value; when
    nothing is there, HPS locked onto an empty bin — for a lone sine there is no
    series to lock onto at all — and the lowest peak at or above ``f_min``
    carrying at least ``min_rel_mag`` of the strongest one is the honest answer.
    """
    if len(peak_freqs) == 0:
        return f0
    f = np.asarray(peak_freqs, dtype=float)
    m = np.asarray(peak_mags, dtype=float)
    if f0 > 0:
        nearest = int(np.argmin(np.abs(np.log2(f / f0))))
        if abs(cents(f[nearest], f0)) <= tol_cents:
            return float(f[nearest])
    strong = f[(m >= min_rel_mag * m.max()) & (f >= f_min)]
    return float(strong.min()) if strong.size else f0


def inharmonicity(
    peak_freqs: np.ndarray, peak_mags: np.ndarray, f0: float, max_harmonic: int = 32
) -> float:
    """Magnitude-weighted mean deviation of partials from an ideal series.

    For each partial, ``k = round(f / f0)`` and the deviation is
    ``|f - k·f0| / (k·f0)`` — dimensionless, 0.0 for a perfectly harmonic
    spectrum (saw, square, stacked sines) and growing with FM/metallic
    inharmonicity. Bounded above by 0.5 per partial by construction.
    """
    if f0 <= 0 or len(peak_freqs) == 0:
        return 0.0
    k = np.round(np.asarray(peak_freqs, dtype=float) / f0)
    keep = (k >= 1) & (k <= max_harmonic)
    if not keep.any():
        return 0.0
    f, m, k = np.asarray(peak_freqs)[keep], np.asarray(peak_mags)[keep], k[keep]
    dev = np.abs(f - k * f0) / (k * f0)
    return weighted_mean(dev, m)


def harmonic_energy_ratio(
    peak_freqs: np.ndarray, peak_mags: np.ndarray, f0: float, tol_cents: float = 35.0
) -> float:
    """Share of partial energy that sits on the harmonic series of ``f0``.

    1.0 means one harmonic source explains the whole spectrum; it falls with
    polyphony (the other notes' partials belong to *their* series, not this
    one), with inharmonic partials, and with noise. It is the honesty gauge for
    :func:`inharmonicity`, which is only a single-source measurement.
    """
    if f0 <= 0 or len(peak_freqs) == 0:
        return 0.0
    f = np.asarray(peak_freqs, dtype=float)
    m = np.asarray(peak_mags, dtype=float)
    total = m.sum()
    if total <= _EPS:
        return 0.0
    k = np.maximum(np.round(f / f0), 1.0)
    on_series = np.abs(1200.0 * np.log2(f / (k * f0))) <= tol_cents
    return float(m[on_series].sum() / total)


def cents(f_hi: float, f_lo: float) -> float:
    """Interval between two frequencies, in cents."""
    if f_lo <= 0 or f_hi <= 0:
        return 0.0
    return 1200.0 * float(np.log2(f_hi / f_lo))


def detune_cents(
    peak_freqs: np.ndarray,
    peak_mags: np.ndarray,
    cluster_cents: float = 70.0,
    min_split_ratio: float = 0.5,
) -> float:
    """Estimate unison detuning from the width of split partial clusters.

    A supersaw/unison patch splits every partial into a cluster of near-copies;
    a single oscillator does not. Peaks within ``cluster_cents`` of their
    neighbour are grouped, and the result is the magnitude-weighted mean cents
    span of the groups that contain more than one peak. ``cluster_cents`` stays
    below a semitone so two *different notes* are never fused into one "detuned"
    voice.

    A unison detunes *every* partial, so at least ``min_split_ratio`` of the
    groups must be split for the reading to count; otherwise the result is 0.0.
    That distinguishes a real unison (nearly all partials split) from an
    inharmonic spectrum that happens to drop two sidebands close together (one
    split group in a dozen) — measured on synthesized cases in the tests.

    Two caveats, both deliberate: it under-reads the true spread when voices are
    too close to resolve at the working FFT size, and on polyphonic spans a
    partial of one note can fall inside another note's cluster — so on a dense
    chord the estimate is approximate in both directions and may fail the split
    test entirely and report 0.0. It is evidence of a unison, not a calibrated
    figure.
    """
    if len(peak_freqs) < 2:
        return 0.0
    order = np.argsort(peak_freqs)
    f, m = np.asarray(peak_freqs, dtype=float)[order], np.asarray(peak_mags, dtype=float)[order]
    spans: list[float] = []
    weights: list[float] = []
    n_groups = 0
    start = 0
    for i in range(1, len(f) + 1):
        if i < len(f) and cents(f[i], f[i - 1]) <= cluster_cents:
            continue
        n_groups += 1
        if i - start > 1:  # a split group, not a lone partial
            spans.append(cents(f[i - 1], f[start]))
            weights.append(float(m[start:i].sum()))
        start = i
    if not spans or len(spans) < min_split_ratio * n_groups:
        return 0.0
    return weighted_mean(np.array(spans), np.array(weights))
