#!/usr/bin/env python3
"""Auto-download Open Access PhysioNet Auricle and VitalDB datasets for ICHD.

WARNING: Full raw datasets require ~8 GB of network transfer (~17-18 GB unpacked
on disk) and may take 20-60+ minutes depending on connection speed.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AURICLE_ZIP_URL = "https://physionet.org/content/auricle/get-zip/1.0.0/"
AURICLE_DIR = ROOT / "data" / "raw" / "auricle"
CHECKSUMS_FILE = ROOT / "data" / "CHECKSUMS.txt"


def _verify_checksums() -> bool:
    if not CHECKSUMS_FILE.exists():
        print(f"Warning: {CHECKSUMS_FILE} not found; skipping checksum verification.")
        return True
    data_dir = ROOT / "data"
    mismatches = 0
    lines = [ln.strip() for ln in CHECKSUMS_FILE.read_text().splitlines() if ln.strip()]
    for ln in lines:
        expected_hash, rel_path = ln.split(None, 1)
        target = data_dir / rel_path.strip()
        if not target.exists():
            print(f"  [MISSING] {rel_path}")
            mismatches += 1
            continue
        h = hashlib.sha256()
        with open(target, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected_hash:
            print(f"  [MISMATCH] {rel_path}")
            mismatches += 1
    if mismatches == 0:
        print(f"Verified {len(lines)}/{len(lines)} SHA-256 checksums in data/CHECKSUMS.txt.")
        return True
    print(f"Checksum verification failed with {mismatches} error(s).")
    return False


def download_auricle(force: bool = False) -> None:
    if (AURICLE_DIR / "RECORDS").exists() and not force:
        print(f"Auricle already present at {AURICLE_DIR}. Verifying checksums...")
        _verify_checksums()
        return

    AURICLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PhysioNet Auricle v1.0.0 (~7.3 GB ZIP -> ~16 GB unpacked)...")
    print(f"Source: {AURICLE_ZIP_URL}")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "auricle-1.0.0.zip"
        last_pct = -1

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            nonlocal last_pct
            if total_size <= 0:
                return
            pct = int(min(100, (block_num * block_size * 100) // total_size))
            if pct >= last_pct + 5:
                last_pct = pct
                gb_done = (block_num * block_size) / (1024 ** 3)
                gb_tot = total_size / (1024 ** 3)
                print(f"  Progress: {pct:3d}% ({gb_done:.2f} / {gb_tot:.2f} GB)", flush=True)

        urllib.request.urlretrieve(AURICLE_ZIP_URL, zip_path, reporthook=_progress)
        print("Extracting Auricle archive...")
        extract_tmp = Path(tmpdir) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_tmp)

        # PhysioNet ZIPs wrap files in a single top-level directory
        entries = list(extract_tmp.iterdir())
        src_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else extract_tmp
        for item in src_root.iterdir():
            dest = AURICLE_DIR / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))

    _verify_checksums()


def download_vitaldb() -> None:
    try:
        import vitaldb  # noqa: F401
    except ImportError:
        print(
            "Error: `vitaldb` package is required for VitalDB download.\n"
            "Install it with: pip install -e \".[external]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    from ichd.external import SELECTED_CASES, cache_case_raw

    print(f"Caching {len(SELECTED_CASES)} VitalDB surgical cases (~791 MB total)...")
    ok_count = 0
    for idx, cid in enumerate(SELECTED_CASES, 1):
        ok = cache_case_raw(cid)
        status = "OK" if ok else "SKIPPED (<30 min or unavailable)"
        print(f"  [{idx:2d}/{len(SELECTED_CASES)}] case_{cid:04d}: {status}")
        if ok:
            ok_count += 1
    print(f"VitalDB caching complete: {ok_count}/{len(SELECTED_CASES)} cases ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Open Access datasets for ICHD (~18 GB total).")
    parser.add_argument(
        "--dataset",
        choices=["auricle", "vitaldb", "all"],
        default="all",
        help="Which dataset to download (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download Auricle even if present.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip interactive confirmation prompt.")
    args = parser.parse_args()

    print("=" * 74)
    print("ICHD DATASET DOWNLOAD SUMMARY")
    print("  1. PhysioNet Auricle v1.0.0 (Open Access, ODC-By):")
    print("     - 70 subjects (107 WFDB sessions @ 1 kHz: ECG, SCG, Ear/Finger PPG, CNAP)")
    print("     - Download: 7.3 GB ZIP -> 16.0 GB unpacked in data/raw/auricle/")
    print("  2. VitalDB Surgical Cohort (Open Access via vitaldb API):")
    print("     - 30 pre-registered OR cases (1 kHz ECG_II, PLETH, FloTrac SV)")
    print("     - Cache size: ~791 MB in data/interim/vitaldb_cache/")
    print("  Total disk footprint: ~17-18 GB (may take 20-60+ minutes)")
    print("=" * 74)

    if not args.yes:
        ans = input("Proceed with download? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    if args.dataset in ("auricle", "all"):
        download_auricle(force=args.force)
    if args.dataset in ("vitaldb", "all"):
        download_vitaldb()


if __name__ == "__main__":
    main()
