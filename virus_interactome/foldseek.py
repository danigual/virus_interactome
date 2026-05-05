"""
foldseek_client.py
------------------
Standalone client for submitting structural homology searches to the
Foldseek web API (https://search.foldseek.com).

Usage example
-------------
    client = FoldseekClient()
    results_tsv = client.search(
        cif_path="my_structure.cif",
        databases=["afdb-swissprot", "pdb100"],
        out_dir="foldseek_results",
    )
    print(results_tsv.read_text())
"""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import glob

class FoldseekClient:
    """
    Client for the Foldseek structural homology search web API.

    Parameters
    ----------
    poll_interval : int, optional
        Seconds between status-check requests while waiting for a job to
        finish. Defaults to 10.
    timeout : int, optional
        Maximum total wait time in seconds before raising ``TimeoutError``.
        Defaults to 600 (10 minutes).

    Examples
    --------
    >>> client = FoldseekClient()
    >>> tsv_path = client.search(
    ...     cif_path="structure.cif",
    ...     databases=["afdb-swissprot", "pdb100"],
    ...     out_dir="results/",
    ... )
    """

    API_BASE = "https://search.foldseek.com/api"

    def __init__(self, poll_interval: int = 10, plddt_threshold: float = 0, timeout: int = 600) -> None:
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._requests = self._import_requests()
        if plddt_threshold < 0 or plddt_threshold > 100:
            raise ValueError("pLDDT threshold must be between 0 and 100.")
        self.plddt_threshold = plddt_threshold  

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        cif_path: str | Path,
        databases: List[str],
        out_dir: str | Path = ".",
        mode: str = "3diaa",
        protein_id: Optional[str] = None,
    ) -> Path:
        """
        Run a full Foldseek search: submit → poll → download.

        Parameters
        ----------
        cif_path : str or Path
            Path to the CIF structure file to search with.
        databases : list of str
            One or more Foldseek database identifiers, e.g.
            ``["afdb-swissprot", "pdb100"]``.

            Available databases include:
            - ``"afdb-swissprot"``  – AlphaFold DB / Swiss-Prot
            - ``"afdb-proteome"``   – AlphaFold DB proteomes
            - ``"pdb100"``          – PDB (all chains)
            - ``"mgnify_esm30"``    – MGnify ESM metagenomic clusters
            - ``"gmgcl_id"``        – GMGC gene clusters

        out_dir : str or Path, optional
            Directory where the result TSV is written. Created if it does
            not exist. Defaults to the current directory.
        mode : str, optional
            Search mode: ``"3diaa"`` (default) or ``"tmalign"``.
        protein_id : str, optional
            Stem used for the output filename (``{protein_id}.tsv``).
            Defaults to the CIF file's stem.

        Returns
        -------
        Path
            Path to the downloaded TSV result file.

        Raises
        ------
        FileNotFoundError
            If ``cif_path`` does not exist.
        RuntimeError
            On non-200 HTTP responses from the API.
        TimeoutError
            If the job does not finish within ``self.timeout`` seconds.
        """
        cif_path = Path(cif_path)
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ## Check if .tsv output already exists and skip if so
        expected_tsv = glob.glob(str(out_dir)+"/*tsv")
        if expected_tsv:
            print(f"[Foldseek] Using existing result: {expected_tsv[0]}")
            return Path(expected_tsv[0])

        ## Filter structure by pLDDT if threshold is set (0 means no filtering)
        if self.plddt_threshold >0 and self.plddt_threshold <= 100:
            print(f"[Foldseek] Applying pLDDT threshold: {self.plddt_threshold}")

            from moleculekit.molecule import Molecule
            mol = Molecule(str(cif_path))
            ca_plddt = mol.beta[mol.name == "CA"]
            ca_chain = mol.chain[mol.name == "CA"]
            ca_resid = mol.resid[mol.name == "CA"]
            keep_resid = ca_resid[ca_plddt >= self.plddt_threshold]
            keep_chain = ca_chain[ca_plddt >= self.plddt_threshold]

            keep_str = " or ".join(f"(chain {chain} and resid {resid})" for chain, resid in zip(keep_chain, keep_resid))
            mol.filter(f"({keep_str})")
            filtered_cif_path = out_dir / f"{cif_path.stem}_filtered.pdb"
            mol.write(str(filtered_cif_path))   
            cif_path = filtered_cif_path

        if protein_id is None:
            protein_id = cif_path.stem

        cif_content = cif_path.read_text(encoding="utf-8")

        print(f"[Foldseek] Submitting job for '{protein_id}' …")
        ticket_id = self._submit(cif_content, databases, mode)
        print(f"[Foldseek] Ticket: {ticket_id}. Waiting for completion …")

        self._poll(ticket_id)
        print(f"[Foldseek] Job complete. Downloading results …")

        tsv_path = self._download(ticket_id, out_dir, protein_id)
        print(f"[Foldseek] Results written to: {tsv_path}")
        return tsv_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_requests():
        try:
            import requests
            return requests
        except ImportError:
            raise ImportError(
                "The 'requests' package is required. "
                "Install it with:  pip install requests"
            )

    def _submit(
        self,
        cif_content: str,
        databases: List[str],
        mode: str = "3diaa",
    ) -> str:
        """POST the structure to /api/ticket and return the ticket ID."""
        payload: Dict[str, Any] = {"q": cif_content, "mode": mode}
        payload["database[]"] = databases  # requests serialises lists correctly

        response = self._requests.post(
            f"{self.API_BASE}/ticket",
            data=payload,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Foldseek submission failed "
                f"(HTTP {response.status_code}): {response.text}"
            )
        return response.json()["id"]

    def _poll(self, ticket_id: str, poll_interval: Optional[int] = None, timeout: Optional[int] = None) -> None:
        """Block until the job is COMPLETE, or raise on ERROR / timeout."""
        interval = poll_interval if poll_interval is not None else self.poll_interval
        limit = timeout if timeout is not None else self.timeout
        elapsed = 0
        while elapsed < limit:
            resp = self._requests.get(
                f"{self.API_BASE}/ticket/{ticket_id}",
                timeout=30,
            )
            resp.raise_for_status()
            status = resp.json().get("status", "")

            if status == "COMPLETE":
                return
            if status == "ERROR":
                raise RuntimeError(
                    f"Foldseek job {ticket_id} reported an ERROR status."
                )

            time.sleep(interval)
            elapsed += interval

        raise TimeoutError(
            f"Foldseek job {ticket_id} did not complete within {limit}s."
        )

    def _download(
        self,
        ticket_id: str,
        out_dir: Path,
        protein_id: str,
    ) -> Path:
        """Download the result archive and write a single TSV to *out_dir*."""
        resp = self._requests.get(
            f"{self.API_BASE}/result/download/{ticket_id}",
            timeout=120,
            stream=True,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download Foldseek results for '{protein_id}' "
                f"(HTTP {resp.status_code}): {resp.text}"
            )

        tsv_path = out_dir / f"{protein_id}.tsv"
        raw_bytes = resp.content

        # The API returns a tar.gz archive; fall back to plain TSV if needed
        try:
            with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tar:
                tsv_chunks: List[str] = []
                for member in tar.getmembers():
                    if member.name.endswith((".tsv", ".m8")):
                        f = tar.extractfile(member)
                        if f is not None:
                            tsv_chunks.append(f.read().decode("utf-8"))
                content = "\n".join(tsv_chunks)
        except tarfile.TarError:
            content = raw_bytes.decode("utf-8")

        tsv_path.write_text(content, encoding="utf-8")
        return tsv_path
    
    # def _parse_output(self, tsv_path: Path, evalue_cutoff: float, top_n: int) -> List[Dict[str, Any]]:
    #     """Read the TSV result file into a list of dicts."""
    #     import csv

    #     _RAW_COLS = [
    #         "query", "target", "fident", "alnlen", "mismatch",
    #         "gapopen", "qstart", "qend", "tstart", "tend", "evalue", "bits",
    #     ]

    #      # Parse raw TSV
    #     try:
    #         df_raw = pd.read_csv(
    #             tsv_path,
    #             sep="\t",
    #             header=None,
    #             names=_RAW_COLS,
    #             comment="#",
    #         )
    #     except Exception as exc:
    #         logger.warning(f"Could not parse TSV for '{protein_id}': {exc}")
    #         continue

    #     df_filtered = df_raw[df_raw["evalue"] <= evalue_cutoff].copy()
    #     df_top = df_filtered.sort_values("evalue").head(top_n).reset_index(drop=True)
    #     df_top.insert(0, "protein_id", protein_id)
    #     df_top.insert(1, "rank", range(1, len(df_top) + 1))

    #     keep_cols = [
    #         "protein_id", "rank", "target", "fident", "alnlen",
    #         "evalue", "bits", "qstart", "qend", "tstart", "tend",
    #     ]
    #     summary_rows.append(df_top[[c for c in keep_cols if c in df_top.columns]])
    #     logger.info(
    #         f"  {protein_id}: {len(df_top)} hits retained "
    #         f"(e-value ≤ {evalue_cutoff}, top {top_n})."
    #     )

    #     if summary_rows:
    #         summary_df = pd.concat(summary_rows, ignore_index=True)
    #     else:
    #         summary_df = pd.DataFrame(columns=[
    #             "protein_id", "rank", "target", "fident", "alnlen",
    #             "evalue", "bits", "qstart", "qend", "tstart", "tend",
    #         ])

    #     summary_df.to_csv(summary_csv, index=False)
    #     logger.info(f"Foldseek summary saved to {summary_csv}")
    #     return summary_df