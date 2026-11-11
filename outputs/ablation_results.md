# Ablation Study: Team Mama Doain Aku Sukses Ya

| Experiment | N Possessions | Mean Dur (s) | Public Macro-F1 | Public Spell-F1 | Public Alive/Dead F1 |
| --- | --- | --- | --- | --- | --- |
| E0: Baseline Floor (smooth50) | 669 | 7.6 | 0.6207 | 0.3256 | 0.899 |
| E1: ATD Kalman (No CV, No RTS, Raw Proximity) | 1915 | 1.2 | 0.5353 | 0.0027 | 0.707 |
| E2: E1 + CV Ball Candidate Detection | 3920 | 0.6 | 0.5467 | 0.0019 | 0.737 |
| E3: E2 + RTS Backward Smoother | 2964 | 0.9 | 0.5394 | 0.0023 | 0.7339 |
| E4: E3 + Intelligent Dead-Ball Engine | 2934 | 0.8 | 0.5611 | 0.0024 | 0.7273 |
| E5: E4 + Controlled Possession (Anti-Deflection, Argmax) | 2492 | 1.5 | 0.7233 | 0.0626 | 0.9235 |
| E6: Mama Doain Aku Sukses Ya (Full HMM Viterbi) | 358 | 10.5 | 0.7843 | 0.4 | 0.9416 |
