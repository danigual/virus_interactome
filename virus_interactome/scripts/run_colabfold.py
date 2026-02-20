#!/usr/bin/env python3
"""
run_colabfold.py
================
Script para generar el CSV de input de ColabFold y lanzar colabfold_batch
en local usando el paquete virus_interactome.

Uso básico (heterodimeros intra-proteoma):
    python run_colabfold.py \
        --proteome_a proteoma.fasta \
        --input_dir  colabfold_inputs/ \
        --output_dir colabfold_outputs/

Uso inter-proteoma:
    python run_colabfold.py \
        --proteome_a proteoma_a.fasta \
        --proteome_b proteoma_b.fasta \
        --mode inter_pairs \
        --input_dir  colabfold_inputs/ \
        --output_dir colabfold_outputs/
"""

import argparse
import logging
import sys
from pathlib import Path

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
        description="Genera el CSV de ColabFold y lanza colabfold_batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Proteomes
    p.add_argument("--proteome_a", required=True,
                   help="Ruta al FASTA del proteoma A.")
    p.add_argument("--proteome_b", default=None,
                   help="Ruta al FASTA del proteoma B (solo para modo inter).")

    # Interactome mode
    p.add_argument("--mode", default="intra_pairs",
                   choices=["intra_pairs", "inter_pairs", "homomers", "single"],
                   help="Tipo de combinaciones a generar.")

    # Directories
    p.add_argument("--input_dir", required=True,
                   help="Directorio donde se guardará el CSV de input para ColabFold.")
    p.add_argument("--output_dir", required=True,
                   help="Directorio donde ColabFold escribirá los resultados.")

    # ColabFold CSV options
    p.add_argument("--csv_name", default="colabfold_input.csv",
                   help="Nombre del fichero CSV de input generado.")

    # ColabFold runtime options
    p.add_argument("--colabfold_bin",
                   default="/media/DATA/localcolabfold/.pixi/envs/default/bin/colabfold_batch",
                   help="Ruta al binario colabfold_batch en el servidor.")
    p.add_argument("--num_recycle", type=int, default=3,
                   help="Número de reciclajes.")
    p.add_argument("--num_models", type=int, default=5,
                   help="Número de modelos por job.")
    p.add_argument("--model_order", default="1,2,3,4,5",
                   help="Orden de los modelos.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Semilla para reproducibilidad.")
    p.add_argument("--no_amber", action="store_true",
                   help="Desactiva la relajación AMBER.")
    p.add_argument("--no_templates", action="store_true",
                   help="Desactiva el uso de templates estructurales.")
    p.add_argument("--no_gpu_relax", action="store_true",
                   help="Desactiva la relajación en GPU.")
    p.add_argument("--dry_run", action="store_true",
                   help="Muestra el comando sin ejecutarlo.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # 1. Generar CSV de input con InteractomeWriter
    logger.info("=== Paso 1: Generando CSV de input para ColabFold ===")
    writer = InteractomeWriter(args.proteome_a, args.proteome_b)

    metas = writer.write_colabfold_csv(
        output_dir=args.input_dir,
        mode=args.mode,
        csv_name=args.csv_name,
    )
    logger.info(f"CSV generado con {len(metas)} jobs en: {args.input_dir}")

    csv_path = Path(args.input_dir) / args.csv_name

    # 2. Lanzar ColabFold con InteractomeRunner
    logger.info("=== Paso 2: Lanzando ColabFold ===")
    runner = InteractomeRunner(
        path_of_inputs=args.input_dir,
        path_of_outputs=args.output_dir,
        mode="colabfold",
    )

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
        dry_run=args.dry_run,
    )

    # 3. Informe final
    status = result.get("status", "UNKNOWN")
    elapsed = result.get("elapsed_s", 0.0)
    log_path = result.get("log_path")

    logger.info(f"Estado final: {status} ({elapsed}s)")
    if log_path:
        logger.info(f"Log de ColabFold: {log_path}")

    if result.get("returncode", 1) != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
