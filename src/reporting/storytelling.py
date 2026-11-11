"""Automated Data Storytelling & Visual Match Report Generator.

Transforms frame-level possession predictions (H, A, D, X) into an accessible
analytical narrative, interactive summary tables, visual charts (PNG), Markdown,
and PDF reports without using ground-truth labels or video feedback.
"""

import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "cache", "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import evaluation.possession_metrics as pm

FPS_DEFAULT = 25.0


def _display_team_name(raw_name, fallback):
    """Normalize club display names cleanly (fixes Metrica XML typo)."""
    name = (raw_name or fallback).strip()
    if "atletic" in name.lower():
        return "Athletic Club"
    if name.lower() == "barcelona":
        return "FC Barcelona"
    return name


def _team_spell_stats(spells, team_code, frames, alive_frames, graded_frames, fps):
    """Compute possession metrics for one team (H or A)."""
    team_durs = np.array([(e - s) / fps for s, e, k in spells if k == team_code], dtype=np.float64)
    n_spells = int(len(team_durs))

    if n_spells > 0:
        mean_dur = float(np.mean(team_durs))
        median_dur = float(np.median(team_durs))
        max_dur = float(np.max(team_durs))
        q_cnt = int(np.sum(team_durs < 5.0))
        m_cnt = int(np.sum((team_durs >= 5.0) & (team_durs < 15.0)))
        s_cnt = int(np.sum(team_durs >= 15.0))
    else:
        mean_dur = median_dur = max_dur = 0.0
        q_cnt = m_cnt = s_cnt = 0

    denom = max(1, n_spells)
    return {
        "frames": int(frames),
        "seconds": float(frames / fps),
        "minutes": float(frames / fps / 60.0),
        "alive_share_pct": float(100.0 * frames / alive_frames) if alive_frames else 0.0,
        "match_pct": float(100.0 * frames / graded_frames) if graded_frames else 0.0,
        "n_spells": n_spells,
        "mean_dur_s": mean_dur,
        "median_dur_s": median_dur,
        "max_dur_s": max_dur,
        "buckets": {
            "quick_lt5": q_cnt,
            "quick_pct": float(100.0 * q_cnt / denom) if n_spells else 0.0,
            "medium_5_15": m_cnt,
            "medium_pct": float(100.0 * m_cnt / denom) if n_spells else 0.0,
            "sustained_ge15": s_cnt,
            "sustained_pct": float(100.0 * s_cnt / denom) if n_spells else 0.0,
        },
    }


def _compute_half_intervals(codes, half_blocks, fps):
    """Compute half-by-half totals and 15-minute tactical momentum intervals."""
    halves = []
    intervals_15m = []
    frames_15m = int(round(15 * 60 * fps))

    for half_num, sf, ef in half_blocks:
        stop = ef + 1
        w = codes[sf:stop]
        hw = int(np.sum(w == pm.H))
        aw = int(np.sum(w == pm.A))
        dw = int(np.sum(w == pm.D))
        alw = hw + aw

        halves.append({
            "half": int(half_num),
            "start_frame": int(sf),
            "end_frame": int(ef),
            "clock_min": float(len(w) / fps / 60.0),
            "alive_min": float(alw / fps / 60.0),
            "home_frames": hw,
            "away_frames": aw,
            "dead_frames": dw,
            "home_share_pct": float(100.0 * hw / alw) if alw else 0.0,
            "away_share_pct": float(100.0 * aw / alw) if alw else 0.0,
            "dead_pct": float(100.0 * dw / len(w)) if len(w) else 0.0,
        })

        # Split each half into ~15-min slices (up to 3 slices per half cleanly)
        base_min = (half_num - 1) * 45
        slice_specs = [
            (f"B{half_num}: {base_min}-{base_min + 15}'", sf, min(stop, sf + frames_15m)),
            (f"B{half_num}: {base_min + 15}-{base_min + 30}'", min(stop, sf + frames_15m), min(stop, sf + 2 * frames_15m)),
            (f"B{half_num}: {base_min + 30}-{base_min + 45}+'", min(stop, sf + 2 * frames_15m), stop),
        ]
        for label, a, b in slice_specs:
            if a >= b:
                continue
            seg = codes[a:b]
            sh = int(np.sum(seg == pm.H))
            sa = int(np.sum(seg == pm.A))
            sd = int(np.sum(seg == pm.D))
            sal = sh + sa
            h_pct = float(100.0 * sh / sal) if sal else 0.0
            a_pct = float(100.0 * sa / sal) if sal else 0.0
            intervals_15m.append({
                "label": label,
                "half": int(half_num),
                "clock_min": float(len(seg) / fps / 60.0),
                "alive_min": float(sal / fps / 60.0),
                "home_share_pct": h_pct,
                "away_share_pct": a_pct,
                "dead_pct": float(100.0 * sd / len(seg)) if len(seg) else 0.0,
                "dominant": "H" if sh > sa else ("A" if sa > sh else "TIE"),
            })

    return halves, intervals_15m


def compute_storytelling_stats(codes, half_blocks, home_name="Barcelona", away_name="Atletic Club", fps=FPS_DEFAULT):
    """Aggregate comprehensive match analytics and narrative metrics from label codes."""
    home_disp = _display_team_name(home_name, "Home")
    away_disp = _display_team_name(away_name, "Away")

    runs = [(sf, ef + 1) for _, sf, ef in half_blocks]
    spells = pm.possession_spells(codes, runs)

    # Count exclusively inside graded half blocks for clean match metrics
    graded_mask = np.zeros(len(codes), dtype=bool)
    for _, sf, ef in half_blocks:
        graded_mask[sf:ef + 1] = True

    graded_codes = codes[graded_mask]
    h_frames = int(np.sum(graded_codes == pm.H))
    a_frames = int(np.sum(graded_codes == pm.A))
    d_frames = int(np.sum(graded_codes == pm.D))
    x_frames = int(len(codes) - len(graded_codes))
    graded_frames = int(len(graded_codes))
    alive_frames = h_frames + a_frames

    h_stats = _team_spell_stats(spells, pm.H, h_frames, alive_frames, graded_frames, fps)
    a_stats = _team_spell_stats(spells, pm.A, a_frames, alive_frames, graded_frames, fps)
    h_stats["name"] = home_disp
    a_stats["name"] = away_disp

    halves, intervals_15m = _compute_half_intervals(codes, half_blocks, fps)

    # Top 5 longest uninterrupted possession spells
    top_spells = []
    for s, e, k in sorted(spells, key=lambda x: x[1] - x[0], reverse=True)[:5]:
        half_idx = 1
        base_sf = half_blocks[0][1]
        for h_num, sf, ef in half_blocks:
            if sf <= s <= ef:
                half_idx = h_num
                base_sf = sf
                break
        minute_mark = (s - base_sf) / fps / 60.0 + (half_idx - 1) * 45.0
        top_spells.append({
            "team_code": "H" if k == pm.H else "A",
            "team_name": home_disp if k == pm.H else away_disp,
            "half": int(half_idx),
            "approx_minute": float(minute_mark),
            "duration_s": float((e - s) / fps),
            "start_frame": int(s),
            "end_frame": int(e),
        })

    diff_pct = abs(round(h_stats["alive_share_pct"], 2) - round(a_stats["alive_share_pct"], 2))
    diff_min = abs(round(h_stats["minutes"], 2) - round(a_stats["minutes"], 2))
    if h_frames > a_frames:
        winner_code = "H"
        winner_name = home_disp
        loser_name = away_disp
    elif a_frames > h_frames:
        winner_code = "A"
        winner_name = away_disp
        loser_name = home_disp
    else:
        winner_code = "TIE"
        winner_name = "Imbang"
        loser_name = ""

    return {
        "home_name": home_disp,
        "away_name": away_disp,
        "overview": {
            "total_frames": int(len(codes)),
            "graded_frames": graded_frames,
            "ungraded_frames": x_frames,
            "total_match_min": float(graded_frames / fps / 60.0),
            "alive_frames": alive_frames,
            "alive_min": float(alive_frames / fps / 60.0),
            "alive_pct": float(100.0 * alive_frames / graded_frames) if graded_frames else 0.0,
            "dead_frames": d_frames,
            "dead_min": float(d_frames / fps / 60.0),
            "dead_pct": float(100.0 * d_frames / graded_frames) if graded_frames else 0.0,
            "total_spells": int(len(spells)),
            "overall_mean_dur_s": float(alive_frames / fps / len(spells)) if spells else 0.0,
        },
        "teams": {"H": h_stats, "A": a_stats},
        "halves": halves,
        "intervals_15m": intervals_15m,
        "top_spells": top_spells,
        "verdict": {
            "winner_code": winner_code,
            "winner_name": winner_name,
            "loser_name": loser_name,
            "margin_pct": float(diff_pct),
            "margin_min": float(diff_min),
        },
    }


def load_optional_public_report(eval_report_path):
    """Load public evaluation JSON report if present on disk."""
    if not eval_report_path or not os.path.isfile(eval_report_path):
        return None
    with open(eval_report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_storytelling_charts(stats, chart_output_path):
    """Render 2x2 visual match storytelling dashboard PNG using matplotlib."""
    os.makedirs(os.path.dirname(chart_output_path), exist_ok=True)

    c_home = "#1A365D"
    c_away = "#C53030"
    c_dead = "#A0AEC0"
    c_grid = "#E2E8F0"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    h_st = stats["teams"]["H"]
    a_st = stats["teams"]["A"]
    ov = stats["overview"]
    h_name = stats["home_name"]
    a_name = stats["away_name"]

    # Panel 1: Match Time & Possession Share
    ax1 = axes[0, 0]
    cats = [f"Bola Hidup\n{h_name} (H)", f"Bola Hidup\n{a_name} (A)", "Bola Mati\n(Dead Ball D)"]
    vals_min = [h_st["minutes"], a_st["minutes"], ov["dead_min"]]
    bars1 = ax1.barh(cats, vals_min, color=[c_home, c_away, c_dead], height=0.55)
    ax1.invert_yaxis()
    ax1.set_xlabel("Durasi (Menit)")
    ax1.set_title(
        f"Distribusi Waktu Laga ({ov['total_match_min']:.1f} Menit Berjalan)\n"
        f"Share Bola Hidup: {h_name} {h_st['alive_share_pct']:.1f}% vs {a_st['alive_share_pct']:.1f}% {a_name}",
        fontsize=10.5, fontweight="bold", color="#1A202C", pad=10
    )
    ax1.grid(axis="x", linestyle="--", alpha=0.5, color=c_grid)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, vals_min):
        ax1.text(val + 0.6, bar.get_y() + bar.get_height() / 2, f"{val:.1f} mnt", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(vals_min) * 1.25 if vals_min else 10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: 15-Minute Match Momentum Swing
    ax2 = axes[0, 1]
    iv = stats["intervals_15m"]
    x_pos = np.arange(len(iv))
    h_shares = [row["home_share_pct"] for row in iv]
    a_shares = [row["away_share_pct"] for row in iv]
    x_labels = [row["label"] for row in iv]

    ax2.plot(x_pos, h_shares, marker="o", linewidth=2.4, color=c_home, label=f"{h_name} (H)")
    ax2.plot(x_pos, a_shares, marker="s", linewidth=2.4, color=c_away, label=f"{a_name} (A)")
    ax2.axhline(50.0, color="#718096", linestyle=":", linewidth=1.2)
    ax2.fill_between(x_pos, h_shares, a_shares, where=np.array(h_shares) >= np.array(a_shares),
                     interpolate=True, alpha=0.12, color=c_home)
    ax2.fill_between(x_pos, h_shares, a_shares, where=np.array(a_shares) > np.array(h_shares),
                     interpolate=True, alpha=0.14, color=c_away)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylim(15, 85)
    ax2.set_ylabel("Possession Share (%)")
    ax2.set_title("Dinamika Momentum per Fase 15 Menit", fontsize=10.5, fontweight="bold", color="#1A202C", pad=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5, color=c_grid)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Possession Style Buckets (Spell Duration Profile)
    ax3 = axes[1, 0]
    bucket_labels = ["Cepat (<5 dtk)", "Sedang (5-15 dtk)", "Build-Up (>=15 dtk)"]
    h_b = [h_st["buckets"]["quick_pct"], h_st["buckets"]["medium_pct"], h_st["buckets"]["sustained_pct"]]
    a_b = [a_st["buckets"]["quick_pct"], a_st["buckets"]["medium_pct"], a_st["buckets"]["sustained_pct"]]
    bx = np.arange(len(bucket_labels))
    w_bar = 0.35

    r1 = ax3.bar(bx - w_bar / 2, h_b, width=w_bar, color=c_home, label=f"{h_name} (Mean {h_st['mean_dur_s']:.1f}s)")
    r2 = ax3.bar(bx + w_bar / 2, a_b, width=w_bar, color=c_away, label=f"{a_name} (Mean {a_st['mean_dur_s']:.1f}s)")
    ax3.set_xticks(bx)
    ax3.set_xticklabels(bucket_labels, fontsize=9)
    ax3.set_ylabel("Proporsi Episode Penguasaan (%)")
    ax3.set_ylim(0, max(h_b + a_b + [45]) * 1.2)
    ax3.set_title("Profil Gaya Penguasaan Bola (% Jumlah Spells)", fontsize=10.5, fontweight="bold", color="#1A202C", pad=10)
    ax3.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5, color=c_grid)
    ax3.set_axisbelow(True)
    for bar in list(r1) + list(r2):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 4: Top 5 Longest Possession Spells
    ax4 = axes[1, 1]
    top = stats["top_spells"]
    if top:
        t_labels = [f"{sp['team_name']} (B{sp['half']} ~{sp['approx_minute']:.0f}')" for sp in top]
        t_durs = [sp["duration_s"] for sp in top]
        t_cols = [c_home if sp["team_code"] == "H" else c_away for sp in top]
        bars4 = ax4.barh(np.arange(len(top)), t_durs, color=t_cols, height=0.55)
        ax4.set_yticks(np.arange(len(top)))
        ax4.set_yticklabels(t_labels, fontsize=8.5)
        ax4.invert_yaxis()
        ax4.set_xlabel("Durasi Tanpa Putus (Detik)")
        ax4.set_xlim(0, max(t_durs) * 1.22)
        for bar, val in zip(bars4, t_durs):
            ax4.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f} dtk", va="center", fontsize=8.5, fontweight="bold")
    ax4.set_title("5 Rangkaian Penguasaan Bola Terpanjang", fontsize=10.5, fontweight="bold", color="#1A202C", pad=10)
    ax4.grid(axis="x", linestyle="--", alpha=0.5, color=c_grid)
    ax4.set_axisbelow(True)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    verdict = stats["verdict"]
    fig.suptitle(
        f"Data Storytelling Penguasaan Bola: {h_name} vs {a_name} — Dominan ke {verdict['winner_name']}",
        fontsize=13.5, fontweight="bold", color="#1A365D", y=0.98
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(chart_output_path, dpi=200)
    plt.close(fig)
    return chart_output_path


def build_narrative_markdown(stats, chart_filename=None, public_metrics=None, sha256=None):
    """Build accessible Indonesian Data Storytelling Markdown report from computed stats."""
    ov = stats["overview"]
    h = stats["teams"]["H"]
    a = stats["teams"]["A"]
    v = stats["verdict"]
    iv = stats["intervals_15m"]

    peak_home = max(iv, key=lambda r: r["home_share_pct"]) if iv else None
    peak_away = max(iv, key=lambda r: r["away_share_pct"]) if iv else None

    lines = [
        f"# Laporan Data Storytelling Penguasaan Bola: {h['name']} vs {a['name']}",
        "",
        f"**Tim Analis:** Mama Doain Aku Sukses Ya  ",
        f"**Kesimpulan Utama:** Bola lebih dominan dan menguntungkan ke pihak **{v['winner_name']} ({v['winner_code']})** "
        f"dengan keunggulan porsi penguasaan **{h['alive_share_pct']:.2f}% vs {a['alive_share_pct']:.2f}%** "
        f"(selisih **{v['margin_pct']:.2f}%** atau **{v['margin_min']:.2f} menit** waktu bola hidup).",
        "",
    ]
    if sha256:
        lines.append(f"**Checksum Submission CSV (SHA-256):** `{sha256}`  ")
        lines.append("")

    if chart_filename:
        lines.extend([
            f"![Dashboard Data Storytelling]({chart_filename})",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 1. Ringkasan Pertandingan dalam Bahasa Sederhana (*Data Storytelling*)",
        "",
        f"- **Waktu Efektif vs Bola Mati:** Dari total **{ov['graded_frames']:,} frame** laga aktif ({ov['total_match_min']:.1f} menit waktu berjalan di Babak 1 & 2, ditambah {ov['ungraded_frames']:,} frame luar laga berlabel `X`), bola benar-benar dimainkan selama **{ov['alive_min']:.2f} menit ({ov['alive_pct']:.1f}%)**. Sisanya **{ov['dead_min']:.2f} menit ({ov['dead_pct']:.1f}%)** adalah kondisi **Bola Mati (`D`)** seperti lemparan ke dalam, tendangan bebas, sepakan sudut, sepakan gawang, selebrasi gol, dan persiapan *kick-off*.",
        f"- **Kontras Gaya Bermain:** Total terjadi **{ov['total_spells']} rangkaian penguasaan bola** (*possession spells*) dengan rata-rata **{ov['overall_mean_dur_s']:.1f} detik**. **{h['name']} (Home)** memegang bola **{h['n_spells']} kali** dengan rata-rata **{h['mean_dur_s']:.2f} detik** per penguasaan (di mana **{h['buckets']['sustained_pct']:.1f}%** di antaranya bertahan lama `>= 15 detik`). Sebaliknya, **{a['name']} (Away)** mencatat **{a['n_spells']} kali** penguasaan dengan rata-rata lebih singkat (**{a['mean_dur_s']:.2f} detik**) dan banyak terhenti cepat (`< 5 detik` sebesar **{a['buckets']['quick_pct']:.1f}%**).",
    ])

    if peak_home and peak_away:
        lines.append(
            f"- **Pergeseran Momentum Pertandingan:** Dominasi **{h['name']}** mencapai puncaknya pada fase **{peak_home['label']}** (**{peak_home['home_share_pct']:.1f}%** vs {peak_home['away_share_pct']:.1f}%). Namun **{a['name']}** sempat bangkit memberikan tekanan balik terkuatnya pada fase **{peak_away['label']}** (**{peak_away['away_share_pct']:.1f}%** vs {peak_away['home_share_pct']:.1f}%)."
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Perbandingan Akhir: Home vs Away",
        "",
        f"| Metrik Lapangan | **Home (`H` — {h['name']})** | **Away (`A` — {a['name']})** | **Bola Mati (`D`)** |",
        "|---|---|---|---|",
        f"| **Jumlah Frame** | {h['frames']:,} | {a['frames']:,} | {ov['dead_frames']:,} |",
        f"| **Durasi Total** | **{h['minutes']:.2f} menit** | {a['minutes']:.2f} menit | {ov['dead_min']:.2f} menit |",
        f"| **Share Bola Hidup (%)** | **{h['alive_share_pct']:.2f}%** | {a['alive_share_pct']:.2f}% | — |",
        f"| **Porsi Total Laga (%)** | {h['match_pct']:.2f}% | {a['match_pct']:.2f}% | {ov['dead_pct']:.2f}% |",
        f"| **Jumlah Penguasaan (*Spells*)** | {h['n_spells']} kali | {a['n_spells']} kali | — |",
        f"| **Rata-rata / Median Durasi** | **{h['mean_dur_s']:.2f} s** / {h['median_dur_s']:.2f} s | {a['mean_dur_s']:.2f} s / {a['median_dur_s']:.2f} s | — |",
        f"| **Penguasaan Terpanjang** | **{h['max_dur_s']:.2f} detik** | {a['max_dur_s']:.2f} detik | — |",
        f"| **Build-Up Panjang (`>= 15s`)** | **{h['buckets']['sustained_ge15']}** ({h['buckets']['sustained_pct']:.1f}%) | {a['buckets']['sustained_ge15']} ({a['buckets']['sustained_pct']:.1f}%) | — |",
        "",
        "### Perbandingan per Babak",
        "",
        "| Babak | Durasi Berjalan | Bola Hidup | Home Share (`H`) | Away Share (`A`) | Bola Mati (`D`) |",
        "|---|---|---|---|---|---|",
    ])

    for hf in stats["halves"]:
        lines.append(
            f"| Babak {hf['half']} | {hf['clock_min']:.1f} mnt | {hf['alive_min']:.1f} mnt | "
            f"**{hf['home_share_pct']:.2f}%** | {hf['away_share_pct']:.2f}% | {hf['dead_pct']:.2f}% |"
        )

    lines.extend([
        "",
        "### Dinamika Momentum per Segmen Waktu (~15 Menit)",
        "",
        "| Segmen Waktu | Bola Hidup | Home (`H`) | Away (`A`) | Bola Mati (`D`) | Tim Dominan |",
        "|---|---|---|---|---|---|",
    ])

    for row in iv:
        dom = h["name"] if row["dominant"] == "H" else (a["name"] if row["dominant"] == "A" else "Imbang")
        lines.append(
            f"| {row['label']} | {row['alive_min']:.1f} mnt | {row['home_share_pct']:.1f}% | "
            f"{row['away_share_pct']:.1f}% | {row['dead_pct']:.1f}% | **{dom}** |"
        )

    lines.extend([
        "",
        "### 5 Rangkaian Penguasaan Bola Terpanjang (*Top Spells*)",
        "",
        "| Peringkat | Tim | Babak | Menit Perkiraan | Durasi | Rentang Frame |",
        "|---|---|---|---|---|---|",
    ])

    for idx, sp in enumerate(stats["top_spells"], 1):
        lines.append(
            f"| {idx} | **{sp['team_name']}** ({sp['team_code']}) | Babak {sp['half']} | "
            f"~menit {sp['approx_minute']:.1f}' | **{sp['duration_s']:.2f} detik** | `{sp['start_frame']}..{sp['end_frame']}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Penjelasan Metrik Evaluasi & Arti Istilah",
        "",
    ])

    if public_metrics:
        trans_25 = public_metrics.get("transitions", {}).get("25", {})
        lines.extend([
            "Berdasarkan evaluasi pada jendela publik 5 menit awal (`7.500` frame, `17` GT possessions):",
            "",
            "| Metrik | Skor Sistem Kita | Arti & Makna Praktis |",
            "|---|---|---|",
            f"| **`macroF1`** | **`{public_metrics.get('macro_f1', 0.0):.4f}`** | Rata-rata tak berbobot skor F1 kelas `H`, `A`, dan `D`. Mengukur ketepatan per frame tanpa bias ke kelas dominan seperti akurasi biasa. |",
            f"| **`spellF1`** | **`{public_metrics.get('spell_f1_mean', 0.0):.4f}`** | **Metrik Peringkat Resmi Lomba.** Mengukur kesesuaian setiap *episode penguasaan utuh* lewat temporal IoU pada ambang `0.3..0.7`. |",
            f"| **`Alive/Dead F1`** | **`{public_metrics.get('alive_dead_f1', 0.0):.4f}`** | Kemampuan model membedakan bola hidup (`H/A`) vs bola mati (`D`). |",
            f"| **`transF1` (±25f)** | **`{trans_25.get('f1', 0.0):.4f}`** | Ketepatan mendeteksi momen perpindahan penguasaan bola dalam toleransi ±1 detik (`lat_f`: `{trans_25.get('mean_latency', 0.0):+.1f}` frame). |",
            f"| **`shareE`** | **`{public_metrics.get('share_err', 0.0):.4f}`** | Selisih absolut porsi penguasaan Home prediksi ({public_metrics.get('pred_home_share', 0.0)*100:.2f}%) vs ground truth ({public_metrics.get('gt_home_share', 0.0)*100:.2f}%). |",
            f"| **`frag`** | **`{public_metrics.get('fragmentation', 0.0):.2f}`** | Rasio jumlah segmen prediksi dibagi asli (ideal mendekati `1.0`; angka tinggi berarti hasil patah-patah). |",
            f"| **`kappa` / `acc`** | `{public_metrics.get('kappa', 0.0):.4f}` / `{public_metrics.get('accuracy', 0.0):.4f}` | Kesesuaian frame terkoreksi peluang acak (*Cohen's Kappa*) dan akurasi per frame. |",
            "",
        ])
    else:
        lines.extend([
            "- **`macroF1`**: Rata-rata tak berbobot dari F1 kelas `H`, `A`, dan `D` pada frame yang dinilai.",
            "- **`spellF1`**: Metrik peringkat utama lomba; mencocokkan episode penguasaan utuh memakai *temporal IoU*.",
            "- **`Alive/Dead F1`**: Ketepatan memisahkan bola aktif (`H/A`) dari bola mati (`D`).",
            "- **`transF1` & `lat_f`**: Akurasi waktu transisi penguasaan (±25 frame) dan jeda keterlambatannya.",
            "- **`shareE` & `frag`**: Galat porsi penguasaan Home dan rasio fragmentasi segmen (ideal ~1.0).",
            "",
        ])

    lines.extend([
        "> **Catatan Metodologi:** Seluruh statistik pertandingan penuh di atas dihitung otomatis dari keluaran inferensi model (karena ground truth penuh bersifat privat), sedangkan jendela publik 5 menit awal digunakan semata-mata sebagai pemeriksaan kewajaran (*sanity check*) sesuai aturan resmi kompetisi.",
        "",
    ])

    return "\n".join(lines)


def generate_storytelling_pdf(stats, chart_path, pdf_output_path, public_metrics=None, sha256=None):
    """Compile visual & narrative Data Storytelling report into a clean PDF document."""
    os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        pdf_output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    c_navy = colors.HexColor("#1A365D")
    c_blue = colors.HexColor("#2B6CB0")
    c_text = colors.HexColor("#2D3748")
    c_hdr = colors.HexColor("#EDF2F7")

    title_st = ParagraphStyle("STitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=c_navy, alignment=1)
    sub_st = ParagraphStyle("SSub", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=c_blue, alignment=1)
    meta_st = ParagraphStyle("SMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=c_text, alignment=1)
    h1_st = ParagraphStyle("SH1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=c_navy, spaceBefore=8, spaceAfter=4)
    body_st = ParagraphStyle("SBody", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=c_text, spaceAfter=4)
    t_cell = ParagraphStyle("STCell", parent=styles["Normal"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=c_text)
    t_hdr = ParagraphStyle("STHdr", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.8, leading=10, textColor=c_navy)

    ov = stats["overview"]
    h = stats["teams"]["H"]
    a = stats["teams"]["A"]
    v = stats["verdict"]

    story = [
        Paragraph("MORE THAN A HACK 2027 — DATA STORYTELLING & HASIL ANALISIS", sub_st),
        Spacer(1, 3),
        Paragraph(f"Laporan Penguasaan Bola: {h['name']} vs {a['name']}", title_st),
        Spacer(1, 3),
        Paragraph(
            f"<b>Tim:</b> Mama Doain Aku Sukses Ya &nbsp;|&nbsp; <b>Dominasi Akhir:</b> {v['winner_name']} "
            f"({h['alive_share_pct']:.2f}% vs {a['alive_share_pct']:.2f}%)",
            meta_st,
        ),
    ]
    if sha256:
        story.append(Paragraph(f"<b>SHA-256 CSV:</b> {sha256}", meta_st))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=c_navy, spaceBefore=2, spaceAfter=6))

    # Storytelling Summary
    story.append(Paragraph("1. Cerita Pertandingan (Data Storytelling)", h1_st))
    story.append(Paragraph(
        f"Dari <b>{ov['graded_frames']:,} frame</b> laga berjalan ({ov['total_match_min']:.1f} menit di Babak 1 & 2), "
        f"waktu bola benar-benar hidup hanya <b>{ov['alive_min']:.2f} menit ({ov['alive_pct']:.1f}%)</b>, sementara "
        f"<b>{ov['dead_min']:.2f} menit ({ov['dead_pct']:.1f}%)</b> merupakan Bola Mati (D). "
        f"Secara keseluruhan terjadi <b>{ov['total_spells']} rangkaian penguasaan bola</b> (rerata {ov['overall_mean_dur_s']:.1f} detik).",
        body_st,
    ))
    story.append(Paragraph(
        f"<b>{h['name']} (Home)</b> lebih mendominasi dan mengontrol jalannya pertandingan dengan <b>{h['minutes']:.2f} menit</b> "
        f"(<b>{h['alive_share_pct']:.2f}%</b> share bola hidup) lewat {h['n_spells']} spells (rerata {h['mean_dur_s']:.2f}s, "
        f"terpanjang <b>{h['max_dur_s']:.2f}s</b>). <b>{a['name']} (Away)</b> mencatat <b>{a['minutes']:.2f} menit</b> "
        f"(<b>{a['alive_share_pct']:.2f}%</b>) lewat {a['n_spells']} spells (rerata {a['mean_dur_s']:.2f}s, "
        f"terpanjang {a['max_dur_s']:.2f}s) dengan proporsi penguasaan singkat &lt;5 detik yang lebih tinggi ({a['buckets']['quick_pct']:.1f}%).",
        body_st,
    ))

    # Dashboard image
    if chart_path and os.path.isfile(chart_path):
        story.append(Spacer(1, 4))
        story.append(Image(chart_path, width=520, height=339))
        story.append(Spacer(1, 4))

    # Head-to-Head Table
    story.append(Paragraph("2. Ringkasan Statistik Head-to-Head & Momentum", h1_st))
    tbl_data = [
        [Paragraph("<b>Indikator</b>", t_hdr), Paragraph(f"<b>Home ({h['name']})</b>", t_hdr), Paragraph(f"<b>Away ({a['name']})</b>", t_hdr), Paragraph("<b>Bola Mati (D)</b>", t_hdr)],
        [Paragraph("Durasi Total / Frame", t_cell), Paragraph(f"<b>{h['minutes']:.2f} mnt</b> ({h['frames']:,})", t_cell), Paragraph(f"{a['minutes']:.2f} mnt ({a['frames']:,})", t_cell), Paragraph(f"{ov['dead_min']:.2f} mnt ({ov['dead_frames']:,})", t_cell)],
        [Paragraph("Possession Share (Bola Hidup)", t_cell), Paragraph(f"<b>{h['alive_share_pct']:.2f}%</b>", t_cell), Paragraph(f"{a['alive_share_pct']:.2f}%", t_cell), Paragraph("-", t_cell)],
        [Paragraph("Jumlah Spells / Rerata Durasi", t_cell), Paragraph(f"{h['n_spells']}x / <b>{h['mean_dur_s']:.2f} dtk</b>", t_cell), Paragraph(f"{a['n_spells']}x / {a['mean_dur_s']:.2f} dtk", t_cell), Paragraph("-", t_cell)],
        [Paragraph("Build-Up Panjang (&ge;15 dtk)", t_cell), Paragraph(f"<b>{h['buckets']['sustained_ge15']}</b> ({h['buckets']['sustained_pct']:.1f}%)", t_cell), Paragraph(f"{a['buckets']['sustained_ge15']} ({a['buckets']['sustained_pct']:.1f}%)", t_cell), Paragraph("-", t_cell)],
    ]
    for hf in stats["halves"]:
        tbl_data.append([
            Paragraph(f"Babak {hf['half']} (Share)", t_cell),
            Paragraph(f"<b>{hf['home_share_pct']:.2f}%</b>", t_cell),
            Paragraph(f"{hf['away_share_pct']:.2f}%", t_cell),
            Paragraph(f"{hf['dead_pct']:.2f}% laga", t_cell),
        ])

    t = Table(tbl_data, colWidths=[150, 130, 130, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_hdr),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 5))

    # Metric Glossary
    story.append(Paragraph("3. Kamus Singkat Metrik Evaluasi", h1_st))
    if public_metrics:
        trans_25 = public_metrics.get("transitions", {}).get("25", {})
        story.append(Paragraph(
            f"<b>Skor Jendela Publik 5 Menit:</b> <b>macroF1 = {public_metrics.get('macro_f1', 0.0):.4f}</b> (rata-rata F1 kelas H/A/D), "
            f"<b>spellF1 = {public_metrics.get('spell_f1_mean', 0.0):.4f}</b> (metrik ranking utama pencocokan episode IoU), "
            f"<b>Alive/Dead F1 = {public_metrics.get('alive_dead_f1', 0.0):.4f}</b>, "
            f"<b>transF1 (±25f) = {trans_25.get('f1', 0.0):.4f}</b>, "
            f"<b>shareE = {public_metrics.get('share_err', 0.0):.4f}</b>, "
            f"<b>frag = {public_metrics.get('fragmentation', 0.0):.2f}</b>.",
            body_st,
        ))
    else:
        story.append(Paragraph(
            "<b>macroF1</b> = rata-rata F1 per kelas (H, A, D); <b>spellF1</b> = metrik peringkat utama berbasis IoU episode penguasaan; "
            "<b>Alive/Dead F1</b> = akurasi bola hidup vs mati; <b>shareE</b> = selisih porsi penguasaan; <b>frag</b> = rasio fragmentasi segmen.",
            body_st,
        ))

    doc.build(story)
    return pdf_output_path


def generate_all_storytelling_artifacts(
    codes,
    half_blocks,
    output_dir,
    team_slug,
    home_name="Barcelona",
    away_name="Atletic Club",
    fps=FPS_DEFAULT,
    sha256=None,
    eval_report_path=None,
):
    """Compute storytelling stats and generate Markdown, Chart PNG, and PDF report files."""
    stats = compute_storytelling_stats(
        codes=codes,
        half_blocks=half_blocks,
        home_name=home_name,
        away_name=away_name,
        fps=fps,
    )
    public_metrics = load_optional_public_report(eval_report_path)

    os.makedirs(output_dir, exist_ok=True)
    chart_name = f"{team_slug}_storytelling.png"
    chart_path = os.path.join(output_dir, chart_name)
    md_path = os.path.join(output_dir, f"{team_slug}_storytelling.md")
    pdf_path = os.path.join(output_dir, f"{team_slug}_storytelling.pdf")

    generate_storytelling_charts(stats, chart_path)

    md_text = build_narrative_markdown(
        stats=stats,
        chart_filename=chart_name,
        public_metrics=public_metrics,
        sha256=sha256,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    generate_storytelling_pdf(
        stats=stats,
        chart_path=chart_path,
        pdf_output_path=pdf_path,
        public_metrics=public_metrics,
        sha256=sha256,
    )

    return {
        "stats": stats,
        "markdown_path": md_path,
        "chart_path": chart_path,
        "pdf_path": pdf_path,
    }


def main():
    import argparse
    import games
    from src.tracking.atd_loader import load_team_mapping
    from src.utils.determinism import compute_file_sha256

    parser = argparse.ArgumentParser(description="Generate Data Storytelling reports from a possession CSV.")
    parser.add_argument("--csv", default="outputs/Mama_Doain_Aku_Sukses_Ya.csv", help="Path to submission CSV")
    parser.add_argument("--out-dir", default="outputs", help="Directory to save storytelling artifacts")
    parser.add_argument("--slug", default="Mama_Doain_Aku_Sukses_Ya", help="Output filename prefix slug")
    parser.add_argument("--game", default=games.GAME, help="Game identifier (default: BAR-ATH)")
    args = parser.parse_args()

    csv_abs = os.path.abspath(args.csv)
    sub_df = pm.read_label_csv(csv_abs)
    codes = pm.label_codes(sub_df)
    half_blocks = games.half_video_blocks(args.game)

    home_name, away_name = "Barcelona", "Atletic Club"
    try:
        (meta_p,) = games.require(["atd_meta"], game=args.game)
        _, home_name, away_name = load_team_mapping(meta_p)
    except OSError:
        pass

    eval_json = os.path.join(PROJECT_ROOT, "evaluation", "reports", f"{args.slug}.json")
    res = generate_all_storytelling_artifacts(
        codes=codes,
        half_blocks=half_blocks,
        output_dir=os.path.abspath(args.out_dir),
        team_slug=args.slug,
        home_name=home_name,
        away_name=away_name,
        fps=float(games.GAMES[args.game]["fps"]),
        sha256=compute_file_sha256(csv_abs),
        eval_report_path=eval_json,
    )
    st = res["stats"]
    print(f"Storytelling generated for {st['home_name']} vs {st['away_name']}:")
    print(f"  Verdict: {st['verdict']['winner_name']} dominates ({st['teams']['H']['alive_share_pct']:.2f}% vs {st['teams']['A']['alive_share_pct']:.2f}%)")
    print(f"  Markdown: {res['markdown_path']}")
    print(f"  Chart PNG: {res['chart_path']}")
    print(f"  PDF: {res['pdf_path']}")


if __name__ == "__main__":
    main()
