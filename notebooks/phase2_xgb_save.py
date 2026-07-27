"""
Fix: Save XGBoost probability map (memory-safe) + generate remaining plots.
The training already completed — model is saved, predictions are in memory.
We just need to reload and save properly.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import joblib, gc, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

OUT = Path(r"D:/1_p3-isro/notebooks")
DFSAR = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA = Path(r"D:/1_p3-isro/datasets/lola")

def ds(arr, f=4):
    return arr[::f, ::f]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading saved XGBoost model...")
model_data = joblib.load(OUT / "xgb_model.joblib")
model = model_data["model"]
scaler = model_data["scaler"]
BAND_NAMES = model_data["bands"]
importances = model.feature_importances_

# ── Load DFSAR bands (one at a time to save memory) ──────────────────────────
BAND_FILES = {
    "CPR": DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif",
    "VOL": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_vol_xx_fp_xx_xxx.tif",
    "EVN": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_evn_xx_fp_xx_xxx.tif",
    "ODD": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_odd_xx_fp_xx_xxx.tif",
    "HLX": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_hlx_xx_fp_xx_xxx.tif",
    "TRT": DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_trt_xx_fp_xx_xxx.tif",
    "SRD": DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_srd_xx_fp_xx_xxx.tif",
}

print("Loading DFSAR bands...")
bands = {}
ref_crs = ref_tf = None
for name, path in BAND_FILES.items():
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if ref_crs is None:
            ref_crs, ref_tf = src.crs, src.transform
            H, W = data.shape
    data[data <= 0] = np.nan
    if name in ("VOL", "EVN", "ODD", "HLX"):
        data = np.log10(data * 1e12)
        data[~np.isfinite(data)] = np.nan
    bands[name] = data
print(f"Grid: {H}x{W}")

# ── Valid mask ────────────────────────────────────────────────────────────────
valid = np.ones((H, W), dtype=bool)
for name in BAND_NAMES:
    valid &= np.isfinite(bands[name])
n_valid = valid.sum()

# ── PSR ───────────────────────────────────────────────────────────────────────
print("Loading PSR...")
with rasterio.open(LOLA / "LPSR_85S_060M_201608.JP2") as src:
    psr_raw = src.read(1).astype(np.float32)
    psr_crs, psr_tf = src.crs, src.transform
psr_binary = ((psr_raw * 0.000025 + 0.5) > 0.5).astype(np.float32)
del psr_raw; gc.collect()
psr_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(psr_binary, psr_on_cpr, src_transform=psr_tf, src_crs=psr_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.nearest)
psr_mask = (psr_on_cpr > 0.5).astype(np.uint8)
del psr_binary, psr_on_cpr; gc.collect()

# ── Slope ─────────────────────────────────────────────────────────────────────
print("Loading DEM...")
with rasterio.open(LOLA / "LM7_final_adj_5mpp_surf.tif") as src:
    dem_raw = src.read(1).astype(np.float32)
    dem_crs, dem_tf = src.crs, src.transform
dem_raw[dem_raw < -9000] = np.nan
dem_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(dem_raw, dem_on_cpr, src_transform=dem_tf, src_crs=dem_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.bilinear)
del dem_raw; gc.collect()
dem_on_cpr[dem_on_cpr == 0] = np.nan
dy, dx = np.gradient(np.where(np.isfinite(dem_on_cpr), dem_on_cpr, 0.0), 25.0)
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope[~np.isfinite(dem_on_cpr)] = np.nan
del dx, dy; gc.collect()

# ── Predict in row-chunks and save directly ──────────────────────────────────
print("Predicting and saving GeoTIFF (row-by-row)...")

out_tif = OUT / "ice_probability_xgb.tif"
with rasterio.open(
    out_tif, "w", driver="GTiff",
    height=H, width=W, count=1, dtype="float32",
    crs=ref_crs, transform=ref_tf, nodata=np.nan,
    compress="deflate"
) as dst:
    ROWS_PER_CHUNK = 200
    for r_start in range(0, H, ROWS_PER_CHUNK):
        r_end = min(r_start + ROWS_PER_CHUNK, H)
        chunk_h = r_end - r_start

        # Extract features for this row chunk
        valid_chunk = valid[r_start:r_end, :]
        n_valid_chunk = valid_chunk.sum()

        if n_valid_chunk == 0:
            row_data = np.full((chunk_h, W), np.nan, dtype=np.float32)
        else:
            features = np.column_stack([bands[name][r_start:r_end, :][valid_chunk] for name in BAND_NAMES])
            features_s = scaler.transform(features)
            proba = model.predict_proba(features_s)[:, 1].astype(np.float32)

            row_data = np.full((chunk_h, W), np.nan, dtype=np.float32)
            row_data[valid_chunk] = proba

            # Apply PSR constraint
            row_data[psr_mask[r_start:r_end, :] == 0] = 0.0

        # Write this chunk
        dst.write(row_data, 1, window=rasterio.windows.Window(0, r_start, W, chunk_h))

        if (r_start // ROWS_PER_CHUNK) % 10 == 0:
            print(f"  Row {r_start}-{r_end} / {H}")

print(f"Saved: {out_tif}")

# ── Reload for plotting (downsampled) ────────────────────────────────────────
print("Loading saved map for plots...")
with rasterio.open(out_tif) as src:
    prob_map_psr = src.read(1)

# ── Phase 1 for comparison ────────────────────────────────────────────────────
cpr = bands["CPR"]
vol = bands["VOL"]
VOL_75 = float(np.nanpercentile(vol[np.isfinite(vol)], 75))
slope_ok = (slope < 20.0) | ~np.isfinite(slope)
phase1_ice = valid & (cpr > 1.0) & (vol > VOL_75) & (psr_mask == 1) & slope_ok
phase1_area = phase1_ice.sum() * 625 / 1e6

ml_ice = prob_map_psr > 0.5
ml_area = ml_ice.sum() * 625 / 1e6
hc_area = (prob_map_psr > 0.7).sum() * 625 / 1e6

ml_only = ml_ice & ~phase1_ice
p1_only = phase1_ice & ~ml_ice
both = ml_ice & phase1_ice

# ── PLOTS ─────────────────────────────────────────────────────────────────────
print("Generating plots...")

# Plot 1: XGBoost vs Threshold
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
fig.suptitle("XGBoost (Mini-RF Labels) vs Thresholds — PSR-Constrained\n"
             "Labels: NASA Mini-RF S-band | Features: ISRO DFSAR L-band | AUC: 0.76",
             fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.imshow(ds(np.clip(cpr, 0, 3)), cmap="inferno", vmin=0, vmax=3, aspect="auto")
p1_show = np.zeros((H, W), dtype=np.uint8); p1_show[phase1_ice] = 255
ax.imshow(np.ma.masked_where(ds(p1_show)==0, ds(p1_show)),
          cmap="cool", vmin=0, vmax=255, alpha=0.85, aspect="auto")
ax.set_title(f"Phase 1: Thresholds\n{phase1_area:.1f} km²")

ax = axes[0, 1]
cmap_prob = plt.cm.RdYlBu_r.copy(); cmap_prob.set_bad("#0d0d1a")
im = ax.imshow(ds(prob_map_psr), cmap=cmap_prob, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="P(ice)", shrink=0.8)
try: ax.contour(ds(psr_mask), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)
except: pass
ax.set_title(f"Phase 2: XGBoost\n{ml_area:.1f} km² (P>0.5)")

ax = axes[1, 0]
agree_map = np.zeros((H, W), dtype=np.uint8)
agree_map[ml_only] = 1; agree_map[p1_only] = 2; agree_map[both] = 3
ax.imshow(ds(agree_map),
          cmap=mcolors.ListedColormap(["#0d0d1a", "#f0a500", "#e94560", "#00d4ff"]),
          vmin=0, vmax=3, aspect="auto")
ax.legend(handles=[
    Patch(facecolor="#00d4ff", label=f"Both agree: {both.sum()*625/1e6:.1f} km²"),
    Patch(facecolor="#f0a500", label=f"XGBoost new: {ml_only.sum()*625/1e6:.1f} km²"),
    Patch(facecolor="#e94560", label=f"Threshold only: {p1_only.sum()*625/1e6:.1f} km²"),
], loc="upper right", fontsize=9)
ax.set_title("Agreement Map")

ax = axes[1, 1]
ax.imshow(ds(np.clip(cpr, 0, 3)), cmap="inferno", vmin=0, vmax=3, aspect="auto")
hc = np.zeros((H, W), dtype=np.uint8); hc[prob_map_psr > 0.7] = 255
ax.imshow(np.ma.masked_where(ds(hc)==0, ds(hc)),
          cmap="winter", vmin=0, vmax=255, alpha=0.85, aspect="auto")
ax.set_title(f"High Confidence (P>0.7)\n{hc_area:.1f} km²")

plt.tight_layout()
plt.savefig(OUT / "08_xgb_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_xgb_vs_threshold.png")

# Plot 2: Feature importance
fig, ax = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(importances)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, 7))
ax.barh([BAND_NAMES[i] for i in sorted_idx], importances[sorted_idx],
        color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Feature Importance")
ax.set_title("XGBoost: Which DFSAR Bands Predict Mini-RF Ice?\n(Cross-sensor learning)", fontsize=13, fontweight="bold")
for i, idx in enumerate(sorted_idx):
    ax.text(importances[idx] + 0.005, i, f"{importances[idx]:.1%}", va="center", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "09_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 09_xgb_feature_importance.png")

print("\n" + "=" * 60)
print("  ALL DONE — XGBoost Results Summary")
print("=" * 60)
print(f"  Phase 1 (thresholds): {phase1_area:.1f} km²")
print(f"  Phase 2 (XGBoost):    {ml_area:.1f} km²")
print(f"  High confidence:      {hc_area:.1f} km²")
print(f"  Both agree:           {both.sum()*625/1e6:.1f} km²")
print(f"  ROC AUC: 0.76")
print(f"\n  Feature importance:")
for name, imp in sorted(zip(BAND_NAMES, importances), key=lambda x: -x[1]):
    print(f"    {name:4s}: {imp:.1%}")
print("=" * 60)
