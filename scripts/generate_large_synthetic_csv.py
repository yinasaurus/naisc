"""
Generate a large synthetic train or test CSV by resampling per-column values from a
reference file (same schema as organiser data).

Example (2M test rows, written in chunks — file stays gitignored as *.csv except train/test):

    python scripts/generate_large_synthetic_csv.py \\
        --reference dataset/test.csv \\
        --output dataset/test_synthetic_2m.csv \\
        --n_rows 2000000 \\
        --chunk_size 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_reference(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def _gen_chunk(
    ref: pd.DataFrame,
    columns: list[str],
    rng: np.random.Generator,
    start_id: int,
    n: int,
) -> pd.DataFrame:
    out: dict[str, np.ndarray | list] = {}
    for c in columns:
        if c == "CustomerID":
            continue
        s = ref[c]
        if pd.api.types.is_numeric_dtype(s):
            col = s.dropna().to_numpy()
            if col.size == 0:
                col = np.array([0.0], dtype=float)
            picked = rng.choice(col, size=n, replace=True)
            if pd.api.types.is_integer_dtype(s):
                picked = np.round(picked).astype(np.int64)
            out[c] = picked
        else:
            vals = s.fillna("").astype(str).unique()
            if vals.size == 0:
                vals = np.array([""])
            out[c] = rng.choice(vals, size=n, replace=True)
    out["CustomerID"] = [f"s{start_id + i:016d}" for i in range(n)]
    return pd.DataFrame(out, columns=columns)


def main() -> None:
    p = argparse.ArgumentParser(description="Chunked synthetic CSV from reference schema")
    p.add_argument("--reference", type=Path, default=Path("dataset/train.csv"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n_rows", type=int, default=2_000_000)
    p.add_argument("--chunk_size", type=int, default=200_000)
    p.add_argument("--ref_sample_rows", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.chunk_size < 1 or args.n_rows < 1:
        raise SystemExit("n_rows and chunk_size must be positive")

    ref = _load_reference(args.reference, args.ref_sample_rows)
    columns = list(ref.columns)
    rng = np.random.default_rng(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    header = True
    while written < args.n_rows:
        n = min(args.chunk_size, args.n_rows - written)
        chunk = _gen_chunk(ref, columns, rng, written, n)
        chunk.to_csv(args.output, mode="w" if header else "a", header=header, index=False)
        written += n
        header = False
        print(f"Wrote {written:,} / {args.n_rows:,} rows -> {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
