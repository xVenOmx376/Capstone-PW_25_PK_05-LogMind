"""Helpers shared by several steps."""
import json
import numpy as np
import pandas as pd
from config import *


def load_templates():
    with open(TEMPLATES_JSON) as f:
        return {int(k): v["template"] for k, v in json.load(f).items()}


def load_sessions():
    return pd.read_pickle(SESSIONS_PKL)


def load_vocab():
    with open(VOCAB_JSON) as f:
        return json.load(f)


def count_matrix(sessions, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(sessions), len(vocab)))
    for r, seq in enumerate(sessions.seq):
        for t in seq:
            if t in idx:
                X[r, idx[t]] += 1
    return X


def feature_matrix(sessions, vocab):
    """log1p template counts + log1p(session length) + log1p(duration). Returns (features, raw counts)."""
    C = count_matrix(sessions, vocab)
    extra = np.column_stack([np.log1p(sessions.n_lines.values), np.log1p(sessions.duration_s.values)])
    return np.hstack([np.log1p(C), extra]), C
