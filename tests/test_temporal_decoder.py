"""Unit tests for HMM Viterbi temporal decoder."""

import pytest
import numpy as np
from src.temporal.hmm_viterbi import (
    viterbi_decode_block,
    decode_full_match,
    DEFAULT_TRANSITION_MATRIX,
    TOTAL_FRAMES,
)


def test_viterbi_flicker_removal():
    """Verify Viterbi eliminates isolated 1-frame noise in a stable sequence."""
    T = 20
    # True state is 0 (H), but frame 10 has a 1-frame noisy emission favoring 1 (A)
    log_emissions = np.full((T, 3), -5.0)
    log_emissions[:, 0] = 0.0  # Strong H
    log_emissions[10, 0] = -2.0
    log_emissions[10, 1] = 0.5  # Noise spike favoring A
    
    log_trans = np.log(DEFAULT_TRANSITION_MATRIX)
    log_init = np.log([0.8, 0.1, 0.1])
    
    path = viterbi_decode_block(log_emissions, log_trans, log_init)
    # The noise spike at t=10 must be rejected; entire path should remain H (0)
    assert (path == 0).all()


def test_decode_full_match_structure():
    """Verify full match output has 199319 rows and 'X' outside half blocks."""
    p_h = np.full(TOTAL_FRAMES, 0.8, dtype=np.float32)
    p_a = np.full(TOTAL_FRAMES, 0.1, dtype=np.float32)
    p_d = np.full(TOTAL_FRAMES, 0.1, dtype=np.float32)
    
    probs = {"p_home": p_h, "p_away": p_a, "p_dead": p_d}
    half_blocks = [(1, 100, 200), (2, 300, 400)]
    
    labels = decode_full_match(probs, half_blocks)
    assert len(labels) == TOTAL_FRAMES
    assert labels[0] == "X"
    assert labels[99] == "X"
    assert labels[100] == "H"
    assert labels[200] == "H"
    assert labels[250] == "X"
    assert labels[300] == "H"

