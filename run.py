"""Main CLI entrypoint for Team 'Mama Doain Aku Sukses Ya'
More Than a Hack 2027: Phase 1 (Ball Possession)
"""

import sys
import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end possession inference pipeline for team 'Mama Doain Aku Sukses Ya'."
    )
    parser.add_argument(
        "--config",
        default="configs/final.yaml",
        help="Path to YAML configuration file (default: configs/final.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save submission CSV (default: outputs/Mama_Doain_Aku_Sukses_Ya.csv)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of CV detection and fusion ignoring cached Parquet files",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip Data Storytelling Markdown/Chart/PDF report generation",
    )
    args = parser.parse_args()

    try:
        results = run_pipeline(
            config_path=args.config,
            output_path=args.output,
            force_recompute=args.force_recompute,
            generate_report=not args.no_report,
        )
        sys.exit(0)
    except Exception as err:
        print(f"\n[FATAL ERROR] Pipeline execution failed: {err}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

