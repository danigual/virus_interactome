#!/usr/bin/env python3
"""
generic_af2_3_inputs.py
=======================
Generic script to generate ColabFold (AF2.3) inputs and launch colabfold_batch
for any viral proteome using the virus_interactome package.

Supports two complementary modes that are run separately:

    1. Heterodimers (all-vs-all intra-proteome pairs):
       python generic_af2_3_inputs.py \\
           --proteome_a TsV-N1_corto_clean.fasta \\
           --mode intra_pairs \\
           --input_dir cf_inputs/ \\
           --output_dir cf_outputs/

    2. Homo-k-mers (k = nmin..nmax copies of each protein):
       python generic_af2_3_inputs.py \\
           --proteome_a TsV-N1_corto_clean.fasta \\
           --mode homomers \\
           --nmin 2 --nmax 6 \\
           --input_dir cf_inputs_homo/ \\
           --output_dir cf_outputs_homo/

    3. Inter-proteome pairs (virus vs host):
       python generic_af2_3_inputs.py \\
           --proteome_a virus.fasta \\
           --proteome_b host.fasta \\
           --mode inter_pairs \\
           --input_dir cf_inputs_inter/ \\
           --output_dir cf_outputs_inter/

Strategy: single CSV batch (Strategy B) — one colabfold_batch call processes
all jobs in the CSV. ColabFold reuses MSA results across jobs sharing the same
sequence, which is significantly faster for large interactomes.
"""

import argparse
import logging
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from virus_interactome import InteractomeWriter, InteractomeRunner

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate ColabFold (AF2.3) CSV inputs and launch colabfold_batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Proteomes
    p.add_argument("--proteome_a", required=True,
                   help="Path to FASTA file of proteome A (virus).")
    p.add_argument("--proteome_b", default=None,
                   help="Path to FASTA file of proteome B (e.g. host). "
                        "Required only for --mode inter_pairs.")

    # Interactome mode
    p.add_argument("--mode", default="intra_pairs",
                   choices=["intra_pairs", "inter_pairs", "homomers", "single"],
                   help="Job generation strategy.")

    # Homomer options (only relevant when --mode homomers)
    p.add_argument("--nmin", type=int, default=2,
                   help="Minimum number of copies for homo-k-mers.")
    p.add_argument("--nmax", type=int, default=6,
                   help="Maximum number of copies for homo-k-mers.")

    # Directories
    p.add_argument("--input_dir", required=True,
                   help="Directory where the ColabFold input CSV will be written.")
    p.add_argument("--output_dir", required=True,
                   help="Directory where ColabFold will write results.")

    # CSV name
    p.add_argument("--csv_name", default="colabfold_input.csv",
                   help="Filename for the generated ColabFold input CSV.")

    # ColabFold binary
    p.add_argument(
        "--colabfold_bin",
        default="/media/DATA/localcolabfold/.pixi/envs/default/bin/colabfold_batch",
        help="Path to the colabfold_batch binary on the remote machine.",
    )

    # ColabFold runtime options
    p.add_argument("--num_recycle", type=int, default=3,
                   help="Number of recycling iterations.")
    p.add_argument("--num_models", type=int, default=5,
                   help="Number of models to generate per job.")
    p.add_argument("--model_order", default="1,2,3,4,5",
                   help="Comma-separated model indices.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed for reproducibility.")
    p.add_argument("--no_amber", action="store_true",
                   help="Disable AMBER relaxation.")
    p.add_argument("--no_templates", action="store_true",
                   help="Disable structural templates.")
    p.add_argument("--no_gpu_relax", action="store_true",
                   help="Disable GPU relaxation.")
    p.add_argument("--extra_args", default="",
                   help="Additional flags for colabfold_batch (quoted, "
                        "e.g. '--zip --stop-at-score 85').")
    p.add_argument("--dry_run", action="store_true",
                   help="Generate inputs but do not launch colabfold_batch.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # --- Step 1: Generate ColabFold CSV ---
    logger.info("=== Step 1: Generating ColabFold input CSV ===")
    logger.info(f"  Proteome A : {args.proteome_a}")
    if args.proteome_b:
        logger.info(f"  Proteome B : {args.proteome_b}")
    logger.info(f"  Mode       : {args.mode}")
    if args.mode == "homomers":
        logger.info(f"  Copies     : {args.nmin}..{args.nmax}")

    writer = InteractomeWriter(args.proteome_a, args.proteome_b)

    metas = writer.write_colabfold_csv(
        output_dir=args.input_dir,
        mode=args.mode,
        nmin=args.nmin,
        nmax=args.nmax,
        csv_name=args.csv_name,
    )
    logger.info(f"CSV written with {len(metas)} jobs → {args.input_dir}/{args.csv_name}")

    csv_path = Path(args.input_dir) / args.csv_name

    # --- Step 2: Launch ColabFold ---
    logger.info("=== Step 2: Launching colabfold_batch ===")
    runner = InteractomeRunner(
        path_of_inputs=args.input_dir,
        path_of_outputs=args.output_dir,
        mode="colabfold",
    )

    extra_flags = shlex.split(args.extra_args) if args.extra_args else None

    result = runner.run_colabfold_csv(
        csv_path=str(csv_path),
        output_dir=args.output_dir,
        colabfold_bin=args.colabfold_bin,
        num_recycle=args.num_recycle,
        num_models=args.num_models,
        model_order=args.model_order,
        random_seed=args.random_seed,
        amber=not args.no_amber,
        templates=not args.no_templates,
        use_gpu_relax=not args.no_gpu_relax,
        extra_flags=extra_flags,
        dry_run=args.dry_run,
    )

    # --- Step 3: Report ---
    status = result.get("status", "UNKNOWN")
    elapsed = result.get("elapsed_s", 0.0)
    log_path = result.get("log_path")

    logger.info(f"Final status : {status} ({elapsed:.1f}s)")
    if log_path:
        logger.info(f"ColabFold log: {log_path}")

    if result.get("returncode", 1) != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
