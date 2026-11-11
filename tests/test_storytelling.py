"""Unit tests for automated Data Storytelling analytics and report generation."""

import os
import numpy as np
import evaluation.possession_metrics as pm
from src.reporting.storytelling import (
    compute_storytelling_stats,
    build_narrative_markdown,
    generate_all_storytelling_artifacts,
)


def test_storytelling_stats_computation():
    """Verify possession share, spells, buckets, and verdict calculations on synthetic data."""
    fps = 25.0
    total = 2000
    codes = np.full(total, pm.X, dtype=np.int8)
    # Half 1: 100..699 (600 frames = 24s), Half 2: 1000..1599 (600 frames = 24s)
    half_blocks = [(1, 100, 699), (2, 1000, 1599)]
    codes[100:500] = pm.H     # 400 frames = 16s sustained spell (>=15s)
    codes[500:650] = pm.A     # 150 frames = 6s medium spell
    codes[650:700] = pm.D     # 50 frames = 2s dead
    codes[1000:1300] = pm.H   # 300 frames = 12s medium spell
    codes[1300:1350] = pm.A   # 50 frames = 2s quick spell (<5s)
    codes[1350:1600] = pm.D   # 250 frames = 10s dead

    stats = compute_storytelling_stats(
        codes=codes,
        half_blocks=half_blocks,
        home_name="Barcelona",
        away_name="Atletic Club",
        fps=fps,
    )

    assert stats["home_name"] == "FC Barcelona"
    assert stats["away_name"] == "Athletic Club"
    assert stats["overview"]["graded_frames"] == 1200
    assert stats["overview"]["ungraded_frames"] == 800
    assert stats["overview"]["dead_frames"] == 300
    assert stats["overview"]["total_spells"] == 4

    h = stats["teams"]["H"]
    a = stats["teams"]["A"]
    # H = 700 frames, A = 200 frames, alive = 900 -> 77.78% vs 22.22%
    assert h["frames"] == 700
    assert a["frames"] == 200
    assert np.isclose(h["alive_share_pct"], 700.0 / 9.0, atol=1e-4)
    assert np.isclose(a["alive_share_pct"], 200.0 / 9.0, atol=1e-4)
    assert h["n_spells"] == 2
    assert h["buckets"]["sustained_ge15"] == 1
    assert h["buckets"]["medium_5_15"] == 1
    assert a["buckets"]["quick_lt5"] == 1
    assert stats["verdict"]["winner_code"] == "H"
    assert stats["top_spells"][0]["duration_s"] == 16.0


def test_storytelling_all_dead_ball_edge_case():
    """Verify zero alive possessions does not trigger division-by-zero errors."""
    codes = np.full(1000, pm.X, dtype=np.int8)
    half_blocks = [(1, 100, 499), (2, 600, 899)]
    codes[100:500] = pm.D
    codes[600:900] = pm.D

    stats = compute_storytelling_stats(codes, half_blocks, fps=25.0)
    assert stats["overview"]["total_spells"] == 0
    assert stats["verdict"]["winner_code"] == "TIE"
    md = build_narrative_markdown(stats)
    assert "Ringkasan Pertandingan" in md


def test_storytelling_artifact_generation(tmp_path):
    """Verify Markdown, PNG chart, and PDF report files are cleanly generated."""
    codes = np.full(1200, pm.X, dtype=np.int8)
    half_blocks = [(1, 100, 499), (2, 700, 1099)]
    codes[100:350] = pm.H
    codes[350:450] = pm.A
    codes[450:500] = pm.D
    codes[700:900] = pm.A
    codes[900:1050] = pm.H
    codes[1050:1100] = pm.D

    res = generate_all_storytelling_artifacts(
        codes=codes,
        half_blocks=half_blocks,
        output_dir=str(tmp_path),
        team_slug="test_team",
        home_name="Barcelona",
        away_name="Atletic Club",
        fps=25.0,
        sha256="deadbeef1234",
    )

    assert os.path.isfile(res["markdown_path"])
    assert os.path.isfile(res["chart_path"]) and os.path.getsize(res["chart_path"]) > 5000
    assert os.path.isfile(res["pdf_path"]) and os.path.getsize(res["pdf_path"]) > 2000

