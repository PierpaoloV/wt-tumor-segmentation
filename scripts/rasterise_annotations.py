#!/usr/bin/env python3
"""
Rasterise ASAP XML annotations to TIFF masks at 2.0 µm spacing.

For each XML in --xml_dir (root only, no subdirs), writes:
  <out_dir>/<stem>.tif       — uint8 mask TIFF, values 0–15, EMPTY_VALUE=255
  <out_dir>/<stem>.json      — sidecar: {"labels": [list of mask ints present]}

Fibrose is merged into Regressie (both → int 6) via the overlay order.
Unannotated pixels receive EMPTY_VALUE (255) → pipeline auto-ignores them.

Meant to run INSIDE the pathology-pipeline Docker image where ASAP and
pathology-common are importable.

Usage (inside Docker):
    python3 /home/user/project/scripts/rasterise_annotations.py \\
        --xml_dir   /home/user/data/annotations \\
        --image_dir /home/user/data/images \\
        --out_dir   /home/user/data/masks \\
        --spacing   2.0 \\
        --workers   4
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# These imports are available inside the pipeline Docker image.
try:
    from digitalpathology.image.processing.conversion import create_annotation_mask
except ImportError as exc:
    sys.exit(
        f"Cannot import pathology-common ({exc}). "
        "Run this script inside the pathology-pipeline Docker image."
    )

# Add project src to path so labels.py is importable regardless of install state.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wt_segmentation.labels import DUTCH_TO_INT, EMPTY_VALUE, INT_TO_ENGLISH, ORDER


def _find_image(image_dir: Path, stem: str) -> Path | None:
    """Find the WSI matching an annotation stem (case-insensitive, any extension)."""
    for ext in (".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: glob for partial name matches (some filenames have suffixes)
    matches = list(image_dir.glob(f"{stem}*"))
    return matches[0] if matches else None


def rasterise_one(
    xml_path: Path,
    image_dir: Path,
    out_dir: Path,
    spacing: float,
) -> dict:
    """Rasterise a single XML → TIFF + sidecar JSON.

    Returns a summary dict with the slide stem and which mask ints were written.
    """
    stem = xml_path.stem
    mask_path = out_dir / f"{stem}.tif"
    sidecar_path = out_dir / f"{stem}.json"

    image_path = _find_image(image_dir, stem)
    if image_path is None:
        log.warning("No image found for %s — skipping", stem)
        return {"stem": stem, "status": "skipped", "reason": "image not found"}

    log.info("Rasterising %s", stem)

    # Build the labels dict: Dutch group name → integer.
    # Fibrose and Regressie both map to 6; the overlay ORDER handles priority.
    labels_dict = {
        dutch: mask_int
        for dutch, mask_int in DUTCH_TO_INT.items()
    }

    # Build the ordered list of group names.  Groups at the end of ORDER
    # overwrite groups earlier in the list (highest priority last).
    # Only include groups that exist in our labels dict.
    order = [g for g in ORDER if g in labels_dict]

    try:
        create_annotation_mask(
            image=str(image_path),
            annotation=str(xml_path),
            label_map=labels_dict,
            conversion_order=order,
            conversion_spacing=spacing,
            spacing_tolerance=0.25,
            output_path=str(mask_path),
            strict=False,
            accept_all_empty=True,
        )
    except Exception as exc:
        log.error("Failed to rasterise %s: %s", stem, exc)
        return {"stem": stem, "status": "error", "reason": str(exc)}

    # Determine which mask integers are actually present in the output.
    present_labels = _read_present_labels(mask_path)

    sidecar = {
        "stem": stem,
        "mask_path": str(mask_path),
        "image_path": str(image_path),
        "labels": sorted(present_labels),
        "label_names": {str(i): INT_TO_ENGLISH[i] for i in sorted(present_labels)},
        "spacing_um": spacing,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    log.info("Done %s — classes present: %s", stem, sorted(present_labels))
    return {"stem": stem, "status": "ok", "labels": sorted(present_labels)}


def _read_present_labels(mask_path: Path) -> set[int]:
    """Return the set of unique pixel values in a mask TIFF, excluding EMPTY_VALUE."""
    try:
        import numpy as np
        from digitalpathology.image.io.imagereader import ImageReader

        reader = ImageReader(str(mask_path))
        # Read at the coarsest available level to keep memory low.
        spacing = reader.spacings[-1]
        dims = reader.shapes[reader.spacings.index(spacing)]
        patch = reader.read(spacing=spacing, row=0, col=0,
                            height=dims[0], width=dims[1])
        reader.close()
        unique = set(np.unique(patch).tolist())
        unique.discard(EMPTY_VALUE)  # 0 = unannotated fill, not a supervised class
        return unique
    except Exception as exc:
        log.warning("Could not read back %s to verify labels: %s", mask_path, exc)
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterise ASAP XML annotations to TIFF masks."
    )
    parser.add_argument(
        "--xml_dir", required=True, type=Path,
        help="Directory containing the 108 root-level XML annotation files.",
    )
    parser.add_argument(
        "--image_dir", required=True, type=Path,
        help="Directory containing the WSI image TIFFs.",
    )
    parser.add_argument(
        "--out_dir", required=True, type=Path,
        help="Output directory for mask TIFFs and JSON sidecars.",
    )
    parser.add_argument(
        "--spacing", type=float, default=2.0,
        help="Mask rasterisation spacing in µm (default: 2.0).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel slides to process (default: 4).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Root-level XMLs only (no subdirs — annotations/ subfolders excluded).
    xml_files = sorted(f for f in args.xml_dir.iterdir()
                       if f.suffix.lower() == ".xml" and f.is_file())
    log.info("Found %d XML files in %s", len(xml_files), args.xml_dir)

    if not xml_files:
        sys.exit("No XML files found. Check --xml_dir.")

    results = {"ok": [], "skipped": [], "error": []}

    if args.workers == 1:
        for xml in xml_files:
            r = rasterise_one(xml, args.image_dir, args.out_dir, args.spacing)
            results[r["status"]].append(r["stem"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(rasterise_one, xml, args.image_dir, args.out_dir, args.spacing): xml
                for xml in xml_files
            }
            for future in as_completed(futures):
                r = future.result()
                results[r["status"]].append(r["stem"])

    log.info(
        "Rasterisation complete — ok: %d  skipped: %d  error: %d",
        len(results["ok"]), len(results["skipped"]), len(results["error"]),
    )
    if results["error"]:
        log.warning("Failed slides: %s", results["error"])
    if results["skipped"]:
        log.warning("Skipped slides (no image found): %s", results["skipped"])


if __name__ == "__main__":
    main()
