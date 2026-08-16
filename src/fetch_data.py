"""
Fetch the raw datasets into data/raw/.

  1. BETE-NET database.json  (85 MB) -- 806 DFT electron-phonon calculations
     WITH the full alpha^2 F(omega) spectral function. This is the dataset the
     hypothesis test actually runs on.
     https://github.com/henniggroup/BETE-NET

  2. UCI Superconductivity Dataset -- 21,263 compounds, 81 composition
     features, experimental Tc. Used only for the step-2 sanity baseline.
     https://github.com/RajeevAtla/Superconductivity-Dataset

  3. (optional, --jarvis) JARVIS-DFT supercon_3d -- 1,058 materials with
     lambda, w_log, mu*, and a McMillan-Allen-Dynes Tc.

     NOTE ON JARVIS: its Tc column is COMPUTED from lambda and w_log by the
     Allen-Dynes formula. It is not an independent measurement. Training a
     model on it to study Allen-Dynes breakdown is circular -- the labels are
     the formula. Fetch it for cross-checking lambda values against an
     independent DFT pipeline, not as a source of Tc labels.

Usage
-----
    python src/fetch_data.py
    python src/fetch_data.py --jarvis
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

SOURCES = {
    "bete_database.json":
        "https://raw.githubusercontent.com/henniggroup/BETE-NET/main/database.json",
    "train.csv":
        "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/train.csv",
    "unique_m.csv":
        "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/unique_m.csv",
}


def _hook(blocks, block_size, total):
    if total > 0:
        pct = min(100.0, 100.0 * blocks * block_size / total)
        sys.stdout.write(f"\r    {pct:5.1f}%  of {total / 1e6:.1f} MB")
        sys.stdout.flush()


def fetch(force: bool = False) -> None:
    os.makedirs(RAW, exist_ok=True)
    for name, url in SOURCES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and not force:
            print(f"  {name}: already present "
                  f"({os.path.getsize(dest) / 1e6:.1f} MB)")
            continue
        print(f"  {name}: downloading")
        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        print(f"\r    done ({os.path.getsize(dest) / 1e6:.1f} MB)          ")


def fetch_jarvis() -> None:
    """
    Pull JARVIS-DFT supercon_3d. Requires `pip install jarvis-tools` and
    network access to figshare (blocked in some sandboxes, fine on a laptop).
    """
    try:
        from jarvis.db.figshare import data
    except ImportError:
        print("  jarvis-tools not installed:  pip install jarvis-tools")
        return
    print("  supercon_3d: downloading via jarvis-tools")
    d = data("supercon_3d")
    import pandas as pd
    df = pd.DataFrame(d)
    out = os.path.join(RAW, "jarvis_supercon_3d.csv")
    df.to_csv(out, index=False)
    print(f"    {len(df)} rows -> {out}")
    print(f"    columns: {list(df.columns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jarvis", action="store_true",
                    help="also fetch JARVIS-DFT supercon_3d (needs figshare)")
    ap.add_argument("--force", action="store_true", help="re-download")
    a = ap.parse_args()
    fetch(force=a.force)
    if a.jarvis:
        fetch_jarvis()
