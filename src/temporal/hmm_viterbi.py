"""3-State Hidden Markov Model and Viterbi Sequence Decoder."""

import numpy as np

TOTAL_FRAMES = 199319
H, A, D = 0, 1, 2
STATE_NAMES = np.array(["H", "A", "D"], dtype="<U1")

# Default 3x3 transition matrix reflecting realistic football segment lengths (25 FPS)
# Mean segment duration ~12-15s (300 frames) -> p_stay ~ 0.996
DEFAULT_TRANSITION_MATRIX = np.array([
    # From:  H       A       D
    [0.996, 0.002, 0.002],  # To H
    [0.002, 0.996, 0.002],  # To A
    [0.002, 0.002, 0.996],  # To D
], dtype=np.float64)

# Initial state priors per half: match starts at D before kickoff
DEFAULT_INITIAL_PRIORS = np.array([0.1, 0.1, 0.8], dtype=np.float64)


def viterbi_decode_block(log_emissions, log_transitions, log_initial):
    """Exact log-space Viterbi algorithm for one continuous sequence.
    
    log_emissions: (T, N_STATES)
    log_transitions: (N_STATES, N_STATES), where entry [i, j] is log P(next=j | prev=i)
    log_initial: (N_STATES,)
    """
    T, K = log_emissions.shape
    viterbi = np.zeros((T, K), dtype=np.float64)
    backpointer = np.zeros((T, K), dtype=np.int32)
    
    # Initialize t = 0
    viterbi[0] = log_initial + log_emissions[0]
    
    # Forward recursion
    for t in range(1, T):
        # Broadcast: (K, 1) + (K, K) -> (K, K) [prev_state, curr_state]
        trans_scores = viterbi[t - 1, :, None] + log_transitions
        backpointer[t] = np.argmax(trans_scores, axis=0)
        viterbi[t] = np.max(trans_scores, axis=0) + log_emissions[t]
        
    # Backtrack best path
    best_path = np.zeros(T, dtype=np.int32)
    best_path[-1] = int(np.argmax(viterbi[-1]))
    
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]
        
    return best_path


def decode_full_match(probabilities, half_blocks, transition_matrix=None, initial_priors=None, smoothing_window=60):
    """Run Viterbi decoding independently for each half, setting ungraded frames to 'X'."""
    trans_mat = transition_matrix if transition_matrix is not None else DEFAULT_TRANSITION_MATRIX
    # Normalize rows to sum to 1
    trans_mat = trans_mat / np.sum(trans_mat, axis=1, keepdims=True)
    log_trans = np.log(np.maximum(1e-12, trans_mat))
    
    init_p = initial_priors if initial_priors is not None else DEFAULT_INITIAL_PRIORS
    init_p = init_p / np.sum(init_p)
    log_init = np.log(np.maximum(1e-12, init_p))
    
    # Prepare global label array initialized with 'X'
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    
    p_h = np.maximum(1e-6, probabilities["p_home"])
    p_a = np.maximum(1e-6, probabilities["p_away"])
    p_d = np.maximum(1e-6, probabilities["p_dead"])
    
    probs_3 = np.stack([p_h, p_a, p_d], axis=-1)  # (199319, 3)
    # Log-emission
    log_emiss_all = np.log(probs_3)
    
    for half_num, start_f, end_f in half_blocks:
        block_emiss = log_emiss_all[start_f:end_f + 1]
        best_states = viterbi_decode_block(block_emiss, log_trans, log_init)
        
        # Post-decoding temporal majority filter over smoothing_window to eliminate micro-chatter
        if smoothing_window > 0:
            w = smoothing_window
            n_b = len(best_states)
            h_cum = np.concatenate(([0], np.cumsum(best_states == H)))
            a_cum = np.concatenate(([0], np.cumsum(best_states == A)))
            d_cum = np.concatenate(([0], np.cumsum(best_states == D)))
            
            idx = np.arange(n_b)
            lo = np.maximum(0, idx - w)
            hi = np.minimum(n_b, idx + w + 1)
            
            h_cnt = h_cum[hi] - h_cum[lo]
            a_cnt = a_cum[hi] - a_cum[lo]
            d_cnt = d_cum[hi] - d_cum[lo]
            
            smoothed_states = np.full(n_b, D, dtype=np.int32)
            h_win = (h_cnt >= a_cnt) & (h_cnt >= d_cnt) & (h_cnt > 0)
            a_win = (a_cnt > h_cnt) & (a_cnt >= d_cnt) & (a_cnt > 0)
            smoothed_states[h_win] = H
            smoothed_states[a_win] = A
            best_states = smoothed_states
            
        labels[start_f:end_f + 1] = STATE_NAMES[best_states]
        
    return labels

