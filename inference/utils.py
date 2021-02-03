import tensorflow as tf
import numpy as np
from librosa import load
from easydict import EasyDict

params = EasyDict({
    'sr': 44100,
    'hop_len': 512,
    'fmin': 27.5,
    'bins_per_octave': 48,
    'n_bins': 356,
    'mono': False,
    'win_len': 9,
    'MAX_VAL': 32767
})


def cqt_dual(wav_path):
    y, _ = load(wav_path, params.sr, params.mono)
    inner_cqt = partial(cqt,
                        sr=params.sr,
                        hop_length=params.hop_len,
                        fmin=params.fmin,
                        n_bins=params.n_bins,
                        bins_per_octave=params.bins_per_octave)
    specs = inner_cqt(y[0]), inner_cqt(y[1])
    specs = np.abs(np.stack(specs, axis=-1))  # H, W, C

    return specs


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.ravel()))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.ravel()))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def normalize_np(specs):
    MAX_VAL = params.MAX_VAL
    if specs.ndim == 4:
        max_vals = np.max(specs, axis=(1, 2), keepdims=True)
    elif specs.ndim == 3:
        max_vals = np.max(specs, axis=(0, 1), keepdims=True)

    specs = specs / (max_vals + 1e-8)  # [B, 356, len, 2] 或者 [356, len, 2]
    specs1 = (specs - 0.5) * 2
    specs2 = (specs1 * MAX_VAL).astype(np.int16)
    return specs2
