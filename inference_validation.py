"""
inference_bzone_v2.py  --  BELBCD change inference -> georeferenced detections.gpkg
                           WITH per-detection confidence (prob_mean / prob_max) + area.

Merges the former two steps into one pass:
  1) run the model on before/after tiles      (was: save binary 0/255 PNGs)
  2) reconstruct the raster + vectorize it     (was: detections.py)

The model's PROBABILITY is kept all the way to the polygon: instead of collapsing
each pixel to 0/255 with a hard threshold, the per-pixel probability (0..1) is
preserved, mosaicked into one georeferenced raster, and sampled inside every
detected region to give a confidence value per polygon.

--threshold is now ONLY the candidate cut: pixels at or above it are grouped into
regions to vectorize.  It no longer makes the final yes/no decision -- that is
left to the reviewer, who filters on prob_mean in QGIS.

Normalization is applied inside PairedFolderDataset from CFG.NORM_MEAN / CFG.NORM_STD
(config.py), so it is identical for every model this file runs.

Tile name stem (batch["name"]):  {suffix}_{row}_{col}_{x}_{y}
  row, col : pixel offsets of the tile's top-left corner into the raster grid
  x,   y   : that same corner's real-world coordinates (map units)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.features import shapes as rasterio_shapes, geometry_mask
from rasterio.windows import Window, transform as window_transform
import shapely
from rasterio.warp import transform as warp_transform
from shapely.geometry import shape as shapely_shape, mapping as shapely_mapping
import geopandas as gpd
import pandas as pd
from PIL import Image

from evaluate import load_model          # your existing model loader
from config import CFG                   # NORM_MEAN / NORM_STD live here


# tile name stem: {suffix}_{row}_{col}_{x}_{y}   (optionally trailed by _r<..>_c<..>)
_NAME_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)(?:_r\d+_c\d+)?$")


# --------------------------------------------------------------------------- #
# data loading -- your PairedFolderDataset / get_loader, unchanged
# --------------------------------------------------------------------------- #
class PairedFolderDataset(Dataset):
    """Pairs images by matching filename across the before/after folders."""

    def __init__(self, before_dir, after_dir, img_size, ext=".png"):
        before_names = {f for f in os.listdir(before_dir) if f.lower().endswith(ext)}
        after_names = {f for f in os.listdir(after_dir) if f.lower().endswith(ext)}
        common = sorted(before_names & after_names)

        missing_before = after_names - before_names
        missing_after = before_names - after_names
        if missing_before or missing_after:
            print(f"Warning: {len(missing_before)} file(s) in after_dir with no match "
                  f"in before_dir, {len(missing_after)} file(s) in before_dir with no "
                  f"match in after_dir. These are skipped.")
        if not common:
            raise ValueError("No matching filenames found between before_dir and after_dir.")

        self.before_dir = before_dir
        self.after_dir = after_dir
        self.names = common
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_a = Image.open(os.path.join(self.before_dir, name)).convert("RGB")
        img_b = Image.open(os.path.join(self.after_dir, name)).convert("RGB")
        return {
            "A": self.transform(img_a),
            "B": self.transform(img_b),
            "name": os.path.splitext(name)[0],
        }


def get_loader(args):
    dataset = PairedFolderDataset(
        before_dir=args.before_dir,
        after_dir=args.after_dir,
        img_size=args.img_size,
        ext=args.ext,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


# --------------------------------------------------------------------------- #
# 1. mosaic per-tile PROBABILITY into one georeferenced raster
# --------------------------------------------------------------------------- #
def assemble_prob(tiles, out_tif: Path, tile_size: int,
                  pixel_size: float, crs: str) -> Path:
    """tiles: list of (row, col, x, y, prob_u8) with prob_u8 a HxW uint8 array
    holding probability * 255.  Writes a single-band uint8 raster (0..255)."""
    origins = {(x - col * pixel_size, y + row * pixel_size)
               for row, col, x, y, _ in tiles}
    if len(origins) != 1:                      # tolerate float noise, stop on real disagreement
        xs = sorted({round(o[0], 3) for o in origins})
        ys = sorted({round(o[1], 3) for o in origins})
        if len(xs) > 1 or len(ys) > 1:
            raise ValueError(
                f"tiles disagree on raster origin: x={xs[:3]}..., y={ys[:3]}... "
                f"-> check --tile-size / --pixel-size")
    ulx, uly = next(iter(origins))

    H = max(t[0] for t in tiles) + tile_size
    W = max(t[1] for t in tiles) + tile_size
    transform = from_origin(ulx, uly, pixel_size, pixel_size)

    canvas = np.zeros((H, W), dtype=np.uint8)
    for row, col, _x, _y, prob in tiles:
        if prob.shape != (tile_size, tile_size):
            prob = np.asarray(Image.fromarray(prob).resize(
                (tile_size, tile_size), Image.NEAREST))
        canvas[row:row + tile_size, col:col + tile_size] = prob

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs=crs, transform=transform, nodata=0,
                       compress="deflate", tiled=True) as dst:
        dst.write(canvas, 1)
    print(f"  wrote probability raster {W}x{H} -> {out_tif}")
    return out_tif


def apply_border_pad(mask_path, pad_meters):
    """Zero a pad_meters-thick border on all four sides.  Harmless if 0."""
    if not pad_meters:
        return
    with rasterio.open(mask_path, "r+") as ds:
        px = abs(ds.transform.a)
        b = int(round(pad_meters / px))
        if b <= 0:
            return
        arr = ds.read(1)
        arr[:b, :] = 0; arr[-b:, :] = 0
        arr[:, :b] = 0; arr[:, -b:] = 0
        ds.write(arr, 1)


# --------------------------------------------------------------------------- #
# 2. vectorize the probability raster -> polygons WITH confidence + area
# --------------------------------------------------------------------------- #
def vectorize_with_confidence(prob_path, threshold: float,
                              min_area: Optional[float] = None,
                              connectivity: int = 8,
                              simplify_tolerance_px: Optional[float] = None,
                              extra_cols: Optional[dict] = None) -> gpd.GeoDataFrame:
    """One feature per contiguous region of pixels >= threshold.  For each region
    the probability is sampled INSIDE the polygon to give prob_mean / prob_max."""
    with rasterio.open(prob_path) as src:
        prob = src.read(1)                     # uint8 0..255 (probability * 255)
        transform = src.transform
        crs = src.crs
        pixel_size = abs(transform.a)
    if crs is None:
        raise ValueError(f"{prob_path} has no CRS -- polygons would be in pixel coords.")

    cut = int(round(threshold * 255))
    candidate = (prob >= cut).astype(np.uint8)  # binary mask only drives shape extraction
    tol = simplify_tolerance_px * pixel_size if simplify_tolerance_px else None

    rows = []
    for geom, value in rasterio_shapes(candidate, mask=candidate.astype(bool),
                                       transform=transform, connectivity=connectivity):
        if value == 0:
            continue
        region = shapely_shape(geom)
        if min_area is not None and region.area < min_area:
            continue

        # --- sample probability inside this region (windowed, no extra deps) ---
        minx, miny, maxx, maxy = region.bounds
        r0, c0 = rowcol(transform, minx, maxy)          # top-left pixel
        r1, c1 = rowcol(transform, maxx, miny)          # bottom-right pixel
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, prob.shape[0] - 1), min(c1, prob.shape[1] - 1)
        sub = prob[r0:r1 + 1, c0:c1 + 1]
        win = Window(c0, r0, sub.shape[1], sub.shape[0])
        wt = window_transform(win, transform)
        inside = geometry_mask([shapely_mapping(region)], out_shape=sub.shape,
                               transform=wt, invert=True)    # True = inside polygon
        vals = sub[inside]
        if vals.size == 0:                               # degenerate sliver
            continue
        prob_mean = float(vals.mean()) / 255.0
        prob_max = float(vals.max()) / 255.0

        if tol is not None:
            region = region.simplify(tol, preserve_topology=True)
        xmin, ymin, xmax, ymax = region.bounds

        row = {
            "geometry": region,
            "prob_mean": round(prob_mean, 4),
            "prob_max": round(prob_max, 4),
            "area_desc": round(region.area, 2),          # == QGIS $area, in m^2
            "region_area": region.area,                  # same value, original name
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        }
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)

    review_cols = ["status", "side", "validation"]        # topographer-filled, empty at creation
    base = ["detection_id", "geometry", "prob_mean", "prob_max", "area_desc",
            "region_area", "xmin", "ymin", "xmax", "ymax"]
    extra_keys = list(extra_cols.keys()) if extra_cols else []
    cols = base + extra_keys + review_cols

    if not rows:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf.insert(0, "detection_id", range(1, len(gdf) + 1))
    for c in review_cols:
        gdf[c] = pd.array([None] * len(gdf), dtype="string")
    return gdf[cols]

MAX_DTM_WINDOW_CELLS = 100_000_000

# --------------------------------------------------------------------------- #
# 4. drape detections on DTM (SMB-safe: local temp + copyfile)
# --------------------------------------------------------------------------- #
def drape_on_dtm(gdf: gpd.GeoDataFrame, dtm_path, nodata_fill: float = 0.0) -> gpd.GeoDataFrame:
    """Sets every vertex's Z to the DTM elevation at its XY (nearest neighbour).
    No-coverage vertices get `nodata_fill`; reads the DTM's own declared nodata."""
    if gdf.empty:
        return gdf
    dtm_path = Path(dtm_path)
    if not dtm_path.exists():
        print(f"  warning: DTM not found at {dtm_path}, writing 2D detections")
        return gdf

    xy = shapely.get_coordinates(gdf.geometry.values)
    with rasterio.open(dtm_path) as src:
        x, y = (xy[:, 0], xy[:, 1]) if gdf.crs == src.crs else \
            warp_transform(gdf.crs, src.crs, xy[:, 0], xy[:, 1])
        rows, cols = np.asarray(rowcol(src.transform, x, y))
        r0, c0 = rows.min(), cols.min()
        win = Window(c0, r0, cols.max() + 1 - c0, rows.max() + 1 - r0)
        if win.width * win.height > MAX_DTM_WINDOW_CELLS:
            raise ValueError(
                f"detections span a {win.width}x{win.height} DTM window, over the "
                f"{MAX_DTM_WINDOW_CELLS}-cell cap"
            )
        dtm = src.read(1, window=win, boundless=True, fill_value=src.nodata, masked=True)
        sampled = dtm[rows - r0, cols - c0]
        z = sampled.filled(nodata_fill)
        z[z < -999] = nodata_fill
        missing = int(np.ma.getmaskarray(sampled).sum())
        if missing > len(z) // 2:
            print(f"  warning: {missing}/{len(z)} vertices fall outside the DTM, "
                  f"check the detections CRS ({gdf.crs}) against the DTM's ({src.crs})")

    geoms = shapely.set_coordinates(shapely.force_3d(gdf.geometry.values), np.column_stack([xy, z]))
    return gdf.set_geometry(geoms)

# --------------------------------------------------------------------------- #
# 4. write gpkg (SMB-safe: local temp + copyfile)
# --------------------------------------------------------------------------- #
def write_gpkg(gdf: gpd.GeoDataFrame, out_path: Path, layer: str = "detections"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        gdf.to_file(tmp_path, layer=layer, driver="GPKG")
        shutil.copyfile(tmp_path, out_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    print(f"  wrote {len(gdf)} detection(s) -> {out_path}")


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="BELBCD change inference -> detections.gpkg with confidence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- model / data ---
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--before_dir", required=True)
    p.add_argument("--after_dir", required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--ext", default=".png")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    # --- output ---
    p.add_argument("--out", required=True, help="output .gpkg path")
    p.add_argument("--layer", default="detections")
    # --- vectorization ---
    p.add_argument("--threshold", type=float, default=0.3,
                   help="CANDIDATE cut: group pixels >= this into regions. "
                        "Set low-to-moderate; prob_mean carries the real confidence.")
    p.add_argument("--min-area", type=float, default=2.0,
                   help="drop regions whose true area < this (m^2)")
    p.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    p.add_argument("--simplify-px", type=float, default=2.0,
                   help="simplify tolerance in pixels (0 to disable)")
    # --- georeferencing ---
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--pixel-size", type=float, default=0.25)
    p.add_argument("--crs", default="EPSG:3812")
    p.add_argument("--pad-meters", type=float, default=0.0)
    # --- provenance (optional) ---
    p.add_argument("--model-name", default="belbcd")
    p.add_argument("--year-before", type=int)
    p.add_argument("--year-after", type=int)
    p.add_argument("--keep-prob", action="store_true",
                   help="keep the reconstructed probability raster (.prob.tif)")
    # DTM draping (optional)
    p.add_argument("--dtm-path", default=None,
               help="DTM raster to drape elevation onto detections as Z")
    p.add_argument("--dtm-nodata-fill", type=float, default=0.0,
               help="elevation to use where DTM has no coverage")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# main  (standalone single-model run; the comparison script imports the helpers)
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.out)
    prob_tif = out_path.with_suffix(".prob.tif")

    loader = get_loader(args)
    model = load_model(args.checkpoint, device)
    print("candidate threshold:", args.threshold)

    tiles = []
    skipped = 0
    with torch.no_grad():
        for batch in loader:
            A, B = batch["A"].to(device), batch["B"].to(device)
            out = model(A, B)

            if isinstance(out, (list, tuple)):          # deep supervision -> last head
                out = out[-1]
            if out.shape[1] == 1:                        # (N,1,H,W) -> sigmoid
                prob = out.sigmoid()[:, 0]
            else:                                         # (N,2,H,W) -> softmax, change = ch 1
                prob = out.softmax(1)[:, 1]

            # keep the probability (0..1) as uint8 0..255 -- NO hard threshold here
            prob_u8 = (prob.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()

            for p, name in zip(prob_u8, batch["name"]):
                stem = name[:-4] if name.endswith(".png") else name
                m = _NAME_RE.search(stem)
                if not m:
                    skipped += 1
                    continue
                row, col, x, y = (int(v) for v in m.groups())
                tiles.append((row, col, x, y, p))

    if skipped:
        print(f"  skipped {skipped} tile(s) with unparseable names")
    if not tiles:
        raise SystemExit("no usable tiles -- expected names like "
                         "{suffix}_{row}_{col}_{x}_{y}")
    print(f"  ran model on {len(tiles)} tile(s)")

    print("[1/2] mosaic probability raster")
    assemble_prob(tiles, prob_tif, args.tile_size, args.pixel_size, args.crs)
    apply_border_pad(prob_tif, args.pad_meters)

    print("[2/2] vectorize with confidence")
    simplify = args.simplify_px if args.simplify_px and args.simplify_px > 0 else None
    extra = {"description": f"inference -> {args.model_name}"}
    if args.year_before and args.year_after:
        extra["year_before"] = args.year_before
        extra["year_after"] = args.year_after

    gdf = vectorize_with_confidence(prob_tif, args.threshold,
                                    min_area=args.min_area,
                                    connectivity=args.connectivity,
                                    simplify_tolerance_px=simplify,
                                    extra_cols=extra)
    if args.dtm_path:
        print("  drape DTM elevation")
        gdf = drape_on_dtm(gdf, args.dtm_path, nodata_fill=args.dtm_nodata_fill)
        
    write_gpkg(gdf, out_path, layer=args.layer)

    if len(gdf):
        minx, miny, _, _ = gdf.total_bounds
        inside = 500_000 < minx < 800_000 and 500_000 < miny < 700_000
        print(f"  {len(gdf)} detection(s); bounds {[round(v) for v in gdf.total_bounds]} "
              f"in-BE-range: {inside}")

    if not args.keep_prob:
        prob_tif.unlink(missing_ok=True)


if __name__ == "__main__":
    main()