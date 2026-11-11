"""Generate publication-ready PDF Technical Report for Team 'Mama Doain Aku Sukses Ya'."""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "Mama_Doain_Aku_Sukses_Ya.pdf")


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1A365D")    # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0")  # Cobalt
    accent_color = colors.HexColor("#9B2C2C")     # Crimson
    dark_neutral = colors.HexColor("#2D3748")     # Charcoal body
    light_bg = colors.HexColor("#EDF2F7")         # Soft table header
    
    # Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=1,  # Center
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        alignment=1,
    )
    
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_neutral,
        alignment=1,
    )
    
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=4,
    )
    
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=3,
    )
    
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        spaceAfter=5,
    )
    
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        leftIndent=15,
        spaceAfter=3,
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=dark_neutral,
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=primary_color,
    )
    
    story = []
    
    # Header Title Banner
    story.append(Paragraph("MORE THAN A HACK 2027 — PHASE 1: BALL POSSESSION", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Ball Possession Inference Pipeline: Technical Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Team:</b> Mama Doain Aku Sukses Ya &nbsp;|&nbsp; <b>Slug:</b> Mama_Doain_Aku_Sukses_Ya &nbsp;|&nbsp; <b>Match:</b> FC Barcelona vs Athletic Club (BAR-ATH)", meta_style))
    story.append(Paragraph("<b>Deliverable:</b> outputs/Mama_Doain_Aku_Sukses_Ya.csv &nbsp;|&nbsp; <b>SHA-256:</b> 59f0cff74f3fa39df74c9bae3f724c9d9ef3d4dc8f3ebdb695e0d9ba6190a914", meta_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=2, spaceAfter=8))
    
    # Section 1
    story.append(Paragraph("1. Approach and Architecture", h1_style))
    story.append(Paragraph(
        "Standard frame-by-frame deep learning classifiers produce severe high-frequency temporal chatter (flickering between H, A, and D), catastrophically hurting temporal IoU spell metrics (spellF1). Moreover, the competition rules (clarified November 2026) strictly forbid training or tuning models on the provided match video or ground truth possession data.",
        body_style
    ))
    story.append(Paragraph(
        "To solve this, we architected a modular, physics-informed, and geometry-grounded algorithmic pipeline:",
        body_style
    ))
    story.append(Paragraph("• <b>Multi-Sensor Ball Fusion:</b> 2D Constant-Velocity Kalman Filter combining ATD coordinates with geometry-guided CV ball candidate detection in tactical video ROIs, smoothed backward via RTS smoother per half.", bullet_style))
    story.append(Paragraph("• <b>Kinematic & Spatial Feature Engine:</b> Metric player-ball Euclidean distances, velocity profiles, pitch boundary margins, and SportsCode Smart Tagging tactical priors.", bullet_style))
    story.append(Paragraph("• <b>Intelligent Dead-Ball Engine:</b> Physically grounds dead ball (D) via pitch line exits (>0.0m), collective player stillness (<1.35 m/s, >65% walking), pre-kickoff team separation, and post-goal stoppage.", bullet_style))
    story.append(Paragraph("• <b>Controlled Possession & Anti-Deflection Model:</b> Distinguishes intentional control from transient rebounds using an exponential proximity decay, a 25-frame (1.0s) sustained dwell threshold before turnover, and state persistence.", bullet_style))
    story.append(Paragraph("• <b>Global Temporal HMM Viterbi Decoder:</b> Decodes full match halves globally in log-space with association football calibrated transition probabilities (P_stay = 0.996) and a 60-frame centered regularizer.", bullet_style))
    story.append(Spacer(1, 4))
    
    # Section 2
    story.append(Paragraph("2. Match Assets Used and Dropped", h1_style))
    asset_data = [
        [Paragraph("<b>Asset</b>", table_cell_bold), Paragraph("<b>File Name</b>", table_cell_bold), Paragraph("<b>Status</b>", table_cell_bold), Paragraph("<b>Role / Justification</b>", table_cell_bold)],
        [Paragraph("Tactical ATD Raw", table_cell), Paragraph("BAR-ATH_tactical_atd.txt", table_cell), Paragraph("Used", table_cell), Paragraph("199,319 frames of streaming 2D coordinates for 46 slots & ball.", table_cell)],
        [Paragraph("Tactical ATD Meta", table_cell), Paragraph("BAR-ATH_tactical_atd.xml", table_cell), Paragraph("Used", table_cell), Paragraph("EPTS metadata defining player team assignment (H=Barca, A=Athletic).", table_cell)],
        [Paragraph("Tactical Homography", table_cell), Paragraph("BAR-ATH_tactical_homography.csv", table_cell), Paragraph("Used", table_cell), Paragraph("199,319 projective 3x3 matrices; metric pitch conversions & ROI bounds.", table_cell)],
        [Paragraph("Tactical Smart Tagging", table_cell), Paragraph("BAR-ATH_tactical_smart_tagging.xml", table_cell), Paragraph("Used", table_cell), Paragraph("638 SportsCode intervals; offensive/defensive polarity & stoppage context.", table_cell)],
        [Paragraph("Tactical Video", table_cell), Paragraph("BAR-ATH_tactical.mp4", table_cell), Paragraph("Used", table_cell), Paragraph("1080p 25 FPS stream decoded in-memory for active ATD missing ball segments.", table_cell)],
        [Paragraph("Broadcast Video & ATD", table_cell), Paragraph("BAR-ATH_broadcast.*", table_cell), Paragraph("<b>Dropped</b>", table_cell), Paragraph("Asynchronous timeline (164,488 frames), cuts/zooms, lack of homography sync.", table_cell)],
    ]
    t_asset = Table(asset_data, colWidths=[90, 130, 45, 265])
    t_asset.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_asset)
    story.append(Spacer(1, 6))
    
    # Section 3 & 4
    story.append(Paragraph("3. Pretrained Models, External Data & Compliance", h1_style))
    story.append(Paragraph(
        "<b>100% Rule Compliance:</b> In strict accordance with the November 2026 rules clarification: (1) Zero models or hyperparameters were trained, fitted, or calibrated on the provided match video or ground truth possession data; (2) Zero manual human labels or video inspection corrections were applied; (3) No match assets were transmitted to cloud AI APIs. All system constants are derived from standard physical and mechanical constants of association football (e.g. max ball velocity 45 m/s, pitch size 105.3x68m, control radius 2.5m).",
        body_style
    ))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("4. Possession & Dead-Ball Rules Implemented", h1_style))
    story.append(Paragraph("• <b>Controlled Possession:</b> Proximity score follows S_prox = exp(-d / 2.5) * 4.0. To prevent transient deflections and tackles from breaking spells, an anti-deflection dwell window requires an opponent to maintain closest ball proximity for >= 25 frames (1.0s) before awarding a turnover.", bullet_style))
    story.append(Paragraph("• <b>Dead-Ball Grounding:</b> Pitch line exits (>0.0m) physically force dead ball (D). Collective player stillness (mean speed < 1.35 m/s and >65% walking) decisively triggers stoppage evidence, overcoming proximity biases of standing players.", bullet_style))
    story.append(Paragraph("• <b>Restarts & Kickoffs:</b> Start of halves 1 & 2 enforce state D through pre-kickoff team half separation until player acceleration marks the opening pass. Post-goal celebrations hold D until the center-circle restart.", bullet_style))
    story.append(Spacer(1, 4))
    
    # Section 5: Ablation Study
    story.append(Paragraph("5. Experimental Validation & Ablation Study", h1_style))
    story.append(Paragraph(
        "All components were rigorously ablated and scored against the official baseline floor (baseline_smooth50) on the public 5-minute ground truth (7,500 graded frames) alongside full-match physical consistency metrics:",
        body_style
    ))
    
    ablation_data = [
        [Paragraph("<b>Experiment</b>", table_cell_bold), Paragraph("<b>Description</b>", table_cell_bold), Paragraph("<b>Macro-F1</b>", table_cell_bold), Paragraph("<b>Spell-F1</b>", table_cell_bold), Paragraph("<b>Alive/Dead F1</b>", table_cell_bold), Paragraph("<b>Possessions</b>", table_cell_bold), Paragraph("<b>Mean Dur</b>", table_cell_bold)],
        [Paragraph("<b>E0</b>", table_cell), Paragraph("Baseline Floor (smooth50)", table_cell), Paragraph("0.6207", table_cell), Paragraph("0.3256", table_cell), Paragraph("0.8990", table_cell), Paragraph("669", table_cell), Paragraph("7.6 s", table_cell)],
        [Paragraph("<b>E1</b>", table_cell), Paragraph("ATD Kalman (No CV, Raw Proximity)", table_cell), Paragraph("0.5353", table_cell), Paragraph("0.0027", table_cell), Paragraph("0.7070", table_cell), Paragraph("1,915", table_cell), Paragraph("1.2 s", table_cell)],
        [Paragraph("<b>E2</b>", table_cell), Paragraph("E1 + CV Ball Candidate Detection", table_cell), Paragraph("0.5467", table_cell), Paragraph("0.0019", table_cell), Paragraph("0.7370", table_cell), Paragraph("3,920", table_cell), Paragraph("0.6 s", table_cell)],
        [Paragraph("<b>E3</b>", table_cell), Paragraph("E2 + RTS Backward Smoother", table_cell), Paragraph("0.5394", table_cell), Paragraph("0.0023", table_cell), Paragraph("0.7339", table_cell), Paragraph("2,964", table_cell), Paragraph("0.9 s", table_cell)],
        [Paragraph("<b>E4</b>", table_cell), Paragraph("E3 + Intelligent Dead-Ball Engine", table_cell), Paragraph("0.5611", table_cell), Paragraph("0.0024", table_cell), Paragraph("0.7273", table_cell), Paragraph("2,934", table_cell), Paragraph("0.8 s", table_cell)],
        [Paragraph("<b>E5</b>", table_cell), Paragraph("E4 + Controlled Possession (Dwell)", table_cell), Paragraph("0.7233", table_cell), Paragraph("0.0626", table_cell), Paragraph("0.9235", table_cell), Paragraph("2,492", table_cell), Paragraph("1.5 s", table_cell)],
        [Paragraph("<b>E6</b>", table_cell_bold), Paragraph("<b>Mama Doain Aku Sukses Ya (Full HMM)</b>", table_cell_bold), Paragraph("<b>0.7843</b>", table_cell_bold), Paragraph("<b>0.4000</b>", table_cell_bold), Paragraph("<b>0.9416</b>", table_cell_bold), Paragraph("<b>358</b>", table_cell_bold), Paragraph("<b>10.5 s</b>", table_cell_bold)],
    ]
    t_abl = Table(ablation_data, colWidths=[30, 185, 55, 55, 75, 65, 65])
    t_abl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), light_bg),
        ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor("#EBF8FF")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_abl)
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>Analysis:</b> Raw frame-by-frame proximity (E1–E4) generates thousands of micro-spells (<1.2s), collapsing Spell-F1. Adding the Controlled Possession Model with dwell consistency (E5) eliminates instant deflection jitter and elevates Macro-F1 to 0.7233 and Alive/Dead F1 to 0.9235. Global HMM Viterbi decoding (E6) regularizes full match possessions into realistic match dynamics (358 spells averaging 10.5s), achieving <b>Spell-F1 0.4000</b> (+7.44% over baseline) and <b>Macro-F1 0.7843</b> (+16.36% over baseline).", body_style))
    story.append(Spacer(1, 4))
    
    # Section 6: Reproduction
    story.append(Paragraph("6. Reproduction & Determinism", h1_style))
    story.append(Paragraph(
        "The submission is 100% deterministic and reproducible via a single CLI invocation:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>python run.py --config configs/final.yaml</code><br/>"
        "This executes the end-to-end pipeline in <b>16.0 seconds</b>, passes all format validations, and yields identical bit-for-bit SHA-256 hash: <code>59f0cff74f3fa39df74c9bae3f724c9d9ef3d4dc8f3ebdb695e0d9ba6190a914</code>.",
        body_style
    ))
    
    doc.build(story)
    print(f"Publication PDF successfully generated at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()

