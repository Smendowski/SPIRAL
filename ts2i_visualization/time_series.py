import numpy as np

n = 160
x = np.linspace(0, 40, n)
sine = np.sin(2 * np.pi * 0.12 * x)


def point(s, pos=96):
    a = s.copy()
    a[pos] += 2.0
    return a


def contextual(s, pos=56, w=20):
    a = s.copy()
    a[pos : pos + w] = np.abs(a[pos : pos + w]) + 0.5
    return a


def collective(s, pos=68, w=40):
    a = s.copy()
    a[pos : pos + w] = 0.3
    return a


def level_shift(s, pos=88):
    a = s.copy()
    a[pos:] += 1.2
    return a


def trend(s, start=48, end=120):
    a = s.copy()
    a[start:end] += np.linspace(0, 1.5, end - start)
    return a


def variance(s, pos=56, w=56):
    a = s.copy()
    a[pos : pos + w] += np.random.normal(0, 0.3, w)
    return a


def frequency(s, pos=56, w=64):
    a = s.copy()
    x_seg = np.linspace(0, w / 4, w)
    a[pos : pos + w] = np.sin(2 * np.pi * 0.4 * x_seg)
    return a


def spike(s, pos=112):
    a = s.copy()
    a[pos] += 2.5
    return a


def seasonal(s, pos=72, w=32):
    a = s.copy()
    a[pos : pos + w] = np.linspace(a[pos], a[pos + w - 1], w)
    return a


ANOMALIES = [
    (sine, "Normal Signal", None),
    (point(sine), "Point Anomaly", 96),
    (contextual(sine), "Contextual Anomaly", None),
    (collective(sine), "Collective Anomaly", None),
    (level_shift(sine), "Level Shift", None),
    (trend(sine), "Trend Anomaly", None),
    (variance(sine), "Variance Anomaly", None),
    (frequency(sine), "Frequency Anomaly", None),
    (spike(sine), "Spike Anomaly", 112),
    (seasonal(sine), "Seasonal Anomaly", None),
]
