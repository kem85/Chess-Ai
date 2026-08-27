#!/usr/bin/env python3
"""
Chess-AI Release Packaging Utility
Bundles the complete Chess-AI project with the interactive Web Arena, neural engine,
launchers, models, and tests into clean, distributable release zip archives.
"""

import os
import sys
import shutil
import zipfile
import hashlib
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"

# Patterns to ignore
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".history",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.swp",
    "*.swo",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".swp",
    ".swo",
}


def should_exclude(rel_path: Path, exclude_pth: bool = False) -> bool:
    """Check whether a path should be excluded from the release archive."""
    parts = rel_path.parts
    for part in parts:
        if part in EXCLUDE_DIRS or part.endswith(".egg-info"):
            return True

    name = rel_path.name
    if name in EXCLUDE_FILES or rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    # Exclude previously generated zip packages inside root
    if rel_path.suffix.lower() == ".zip":
        return True

    # Optional exclude of heavy PyTorch .pth for lightweight package
    if exclude_pth and rel_path.suffix.lower() == ".pth":
        return True

    return False


def calculate_sha256(filepath: Path) -> str:
    """Compute the SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def create_release_zip(output_filename: str, exclude_pth: bool = False, copy_to_root: bool = True) -> Path:
    """Creates a zip archive of the entire project."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / output_filename
    root_zip_path = ROOT_DIR / output_filename

    edition_label = " (ONNX-Lite Edition)" if exclude_pth else " (Complete Edition)"
    print(f"\n{'=' * 70}")
    print(f"📦 Packaging Chess-AI Release: {output_filename}{edition_label}")
    print(f"{'=' * 70}")

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
            
            for file in sorted(files):
                abs_file_path = Path(root) / file
                rel_path = abs_file_path.relative_to(ROOT_DIR)

                if should_exclude(rel_path, exclude_pth=exclude_pth):
                    continue

                # Add file to zip under top-level folder "Chess-AI/"
                archive_name = Path("Chess-AI") / rel_path
                file_size = abs_file_path.stat().st_size
                total_uncompressed_bytes += file_size
                total_files += 1

                zipf.write(abs_file_path, arcname=str(archive_name))
                print(f"  [+] {str(rel_path):<48} ({file_size:>10,} bytes)")

    if copy_to_root:
        shutil.copy2(zip_path, root_zip_path)

    # Calculate checksums
    sha256_hash = calculate_sha256(zip_path)
    zip_size = zip_path.stat().st_size
    ratio = (1 - (zip_size / max(total_uncompressed_bytes, 1))) * 100

    print(f"\n{'-' * 70}")
    print(f"✨ {output_filename} Generated Successfully!")
    print(f"  📁 Output:              {zip_path}")
    if copy_to_root:
        print(f"  📁 Root Copy:           {root_zip_path}")
    print(f"  📄 Total Files:         {total_files}")
    print(f"  💾 Uncompressed Size:   {total_uncompressed_bytes / (1024 * 1024):.2f} MB ({total_uncompressed_bytes:,} bytes)")
    print(f"  📦 Compressed Size:     {zip_size / (1024 * 1024):.2f} MB ({zip_size:,} bytes)")
    print(f"  ⚡ Compression Ratio:   {ratio:.1f}% space saved")
    print(f"  🔒 SHA-256 Checksum:    {sha256_hash}")
    print(f"{'-' * 70}\n")

    return zip_path


def build_all_releases(version: str = "1.0.0"):
    """Builds complete and lite packages, plus summary checksums."""
    print(f"\n======================================================================")
    print(f"♟️  CHESS-AI RELEASE BUILDER (v{version})")
    print(f"======================================================================")

    # Clean temporary directories
    for cleanup_dir in [ROOT_DIR / "build", ROOT_DIR / "chess_ai_arena.egg-info"]:
        if cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    # 1. Complete Package (Full ONNX + PyTorch weights)
    full_zip = create_release_zip(f"Chess-AI-v{version}.zip", exclude_pth=False, copy_to_root=True)
    # Also copy as generic Chess-AI.zip in root
    shutil.copy2(full_zip, ROOT_DIR / "Chess-AI.zip")

    # 2. Lite Package (ONNX INT8 engine ~26MB, PyTorch weights auto-downloaded if needed)
    lite_zip = create_release_zip(f"Chess-AI-v{version}-ONNX-Lite.zip", exclude_pth=True, copy_to_root=False)

    # 3. Generate SHA256SUMS.txt
    checksums_file = DIST_DIR / "SHA256SUMS.txt"
    with open(checksums_file, "w", encoding="utf-8") as f:
        for p in sorted(DIST_DIR.glob("*.zip")):
            f.write(f"{calculate_sha256(p)}  {p.name}\n")
        for p in sorted(DIST_DIR.glob("*.whl")):
            f.write(f"{calculate_sha256(p)}  {p.name}\n")
        for p in sorted(DIST_DIR.glob("*.tar.gz")):
            f.write(f"{calculate_sha256(p)}  {p.name}\n")

    shutil.copy2(checksums_file, ROOT_DIR / "SHA256SUMS.txt")

    print("=" * 70)
    print(f"🎉 ALL RELEASE PACKAGES READY IN 'dist/' AND ROOT DIRECTORY!")
    print(f"   - {DIST_DIR / f'Chess-AI-v{version}.zip'}")
    print(f"   - {DIST_DIR / f'Chess-AI-v{version}-ONNX-Lite.zip'}")
    print(f"   - {ROOT_DIR / 'Chess-AI.zip'}")
    print(f"   - {checksums_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ver = "1.0.0"
    if len(sys.argv) > 1:
        ver = sys.argv[1].lstrip("v")
    build_all_releases(ver)
