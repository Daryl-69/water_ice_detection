"""Regenerate web layers using the NEW XGBoost probability map."""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image
import matplotlib.pyplot as plt
import json, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

t0 = time.time()
DFSAR = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA  = Path(r"D:/1_p3-isro/datasets/lola")
OUT   = Path(r"D:/1_p3-isro/site/assets/layers")
OUT.mkdir(parents=True, exist_ok=True)

F = 4

# ── Load CPR ──
print("Loading CPR...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif") as src:
    cpr = src.read(1).astype(np.float32)
    ref_crs, ref_tf = src.crs, src.transform
    H, W = cpr.shape
cpr[cpr <= 0] = np.nan

# ── Load VOL ──
print("Loading VOL...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_vol_xx_fp_xx_xxx.tif") as src:
    vol = src.read(1).astype(np.float32)
vol[vol <= 0] = np.nan
vol_log = np.log10(vol * 1e12)
vol_log[~np.isfinite(vol_log)] = np.nan

# ── Load PSR ──
print("Loading PSR...")
with rasterio.open(LOLA / "LPSR_85S_060M_201608.JP2") as src:
    psr_raw = src.read(1).astype(np.float32)
    psr_crs, psr_tf = src.crs, src.transform
psr_binary = ((psr_raw * 0.000025 + 0.5) > 0.5).astype(np.float32)
psr_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(psr_binary, psr_on_cpr,
          src_transform=psr_tf, src_crs=psr_crs,
          dst_transform=ref_tf, dst_crs=ref_crs,
          resampling=Resampling.nearest)
psr_mask = (psr_on_cpr > 0.5).astype(np.uint8)

# ── Load DEM + slope ──
print("Loading DEM...")
with rasterio.open(LOLA / "LM7_final_adj_5mpp_surf.tif") as src:
    dem_raw = src.read(1).astype(np.float32)
    dem_crs, dem_tf = src.crs, src.transform
dem_raw[dem_raw < -9000] = np.nan
dem_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(dem_raw, dem_on_cpr,
          src_transform=dem_tf, src_crs=dem_crs,
          dst_transform=ref_tf, dst_crs=ref_crs,
          resampling=Resampling.bilinear)
dem_on_cpr[dem_on_cpr == 0] = np.nan
dy, dx = np.gradient(np.where(np.isfinite(dem_on_cpr), dem_on_cpr, 0.0), 25.0)
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope[~np.isfinite(dem_on_cpr)] = np.nan

# ── Load NEW XGBoost probability map ──
print("Loading XGBoost probability map...")
with rasterio.open(Path(r"D:/1_p3-isro/notebooks/ice_probability_xgb.tif")) as src:
    prob_map = src.read(1)

# ── Phase 1 ice mask ──
valid = np.isfinite(cpr) & np.isfinite(vol_log)
VOL_75 = float(np.nanpercentile(vol_log[np.isfinite(vol_log)], 75))
slope_ok = (slope < 20.0) | ~np.isfinite(slope)
phase1_ice = valid & (cpr > 1.0) & (vol_log > VOL_75) & (psr_mask == 1) & slope_ok

def ds(arr):
    return arr[::F, ::F]

h, w = ds(cpr).shape
print(f"Output size: {w} x {h} pixels")

# ═══ Generate layers ═══

# 1. CPR base map
print("Generating CPR layer...")
cpr_ds = ds(np.clip(cpr, 0, 3))
cpr_norm = np.where(np.isfinite(cpr_ds), cpr_ds / 3.0, 0).clip(0, 1)
cmap = plt.cm.inferno
rgba = (cmap(cpr_norm) * 255).astype(np.uint8)
rgba[~np.isfinite(cpr_ds)] = [10, 10, 26, 255]
Image.fromarray(rgba).save(OUT / "cpr_base.png")

# 2. Volume scattering
print("Generating VOL layer...")
vl = vol_log[np.isfinite(vol_log)]
vmin, vmax = float(np.nanpercentile(vl, 2)), float(np.nanpercentile(vl, 98))
vol_ds = ds(vol_log)
vol_norm = np.where(np.isfinite(vol_ds), (vol_ds - vmin) / (vmax - vmin), 0).clip(0, 1)
cmap_v = plt.cm.viridis
rgba = (cmap_v(vol_norm) * 255).astype(np.uint8)
rgba[~np.isfinite(vol_ds)] = [10, 10, 26, 255]
Image.fromarray(rgba).save(OUT / "vol_base.png")

# 3. PSR overlay
print("Generating PSR layer...")
psr_ds = ds(psr_mask)
rgba = np.zeros((h, w, 4), dtype=np.uint8)
rgba[psr_ds == 1] = [26, 95, 180, 160]
Image.fromarray(rgba).save(OUT / "psr_overlay.png")

# 4. Phase 1 ice
print("Generating Phase 1 ice layer...")
p1_ds = ds(phase1_ice.astype(np.uint8))
rgba = np.zeros((h, w, 4), dtype=np.uint8)
rgba[p1_ds == 1] = [0, 212, 255, 220]
Image.fromarray(rgba).save(OUT / "ice_phase1.png")

# 5. XGBoost ML probability (NEW)
print("Generating XGBoost ML probability layer...")
prob_ds = ds(prob_map)
cmap_prob = plt.cm.plasma
rgba = np.zeros((h, w, 4), dtype=np.uint8)
mask = np.isfinite(prob_ds) & (prob_ds > 0.1)
if mask.any():
    colors = cmap_prob(prob_ds[mask].clip(0, 1))
    rgba[mask, 0] = (colors[:, 0] * 255).astype(np.uint8)
    rgba[mask, 1] = (colors[:, 1] * 255).astype(np.uint8)
    rgba[mask, 2] = (colors[:, 2] * 255).astype(np.uint8)
    rgba[mask, 3] = (prob_ds[mask].clip(0.3, 1.0) * 220).astype(np.uint8)
Image.fromarray(rgba).save(OUT / "ice_ml.png")

# 6. Slope safety
print("Generating slope layer...")
slope_ds = ds(slope)
rgba = np.zeros((h, w, 4), dtype=np.uint8)
safe = np.isfinite(slope_ds) & (slope_ds < 15)
caution = np.isfinite(slope_ds) & (slope_ds >= 15) & (slope_ds < 25)
danger = np.isfinite(slope_ds) & (slope_ds >= 25)
rgba[safe] = [39, 174, 96, 180]
rgba[caution] = [243, 156, 18, 180]
rgba[danger] = [192, 57, 43, 180]
Image.fromarray(rgba).save(OUT / "slope_safety.png")

# 7. DEM terrain
print("Generating DEM layer...")
dem_ds = ds(dem_on_cpr)
rgba = np.zeros((h, w, 4), dtype=np.uint8)
dem_valid = np.isfinite(dem_ds)
if dem_valid.any():
    d = dem_ds[dem_valid]
    d_norm = ((d - d.min()) / (d.max() - d.min())).clip(0, 1)
    cmap_t = plt.cm.terrain
    colors = cmap_t(d_norm)
    rgba[dem_valid, 0] = (colors[:, 0] * 255).astype(np.uint8)
    rgba[dem_valid, 1] = (colors[:, 1] * 255).astype(np.uint8)
    rgba[dem_valid, 2] = (colors[:, 2] * 255).astype(np.uint8)
    rgba[dem_valid, 3] = 200
Image.fromarray(rgba).save(OUT / "dem_terrain.png")

# 8. Confidence overlay
print("Generating confidence layer...")
cpr_ice = (cpr > 1.0) & np.isfinite(cpr)
vol_ice = (vol_log > VOL_75) & np.isfinite(vol_log)
in_shadow = psr_mask == 1
conf = np.zeros((H, W), dtype=np.uint8)
conf[cpr_ice & valid] += 1
conf[vol_ice & valid] += 1
conf[in_shadow & valid] += 1
conf[slope_ok & valid] += 1
conf[~valid] = 0
conf_ds = ds(conf)
cmap_conf = plt.cm.cool
rgba = np.zeros((h, w, 4), dtype=np.uint8)
for score in [1, 2, 3, 4]:
    m = conf_ds == score
    if m.any():
        c = cmap_conf(score / 4.0)
        rgba[m] = [int(c[0]*255), int(c[1]*255), int(c[2]*255), int(50 + score * 50)]
Image.fromarray(rgba).save(OUT / "confidence.png")

# ═══ Stats JSON (updated with XGBoost) ═══
print("Generating stats...")
ml_ice_50 = (prob_map > 0.5)
ml_ice_70 = (prob_map > 0.7)
ml_ice_90 = (prob_map > 0.9)

stats = {
    "survey_area_km2": round(np.isfinite(cpr).sum() * 625 / 1e6, 1),
    "psr_area_km2": round(psr_mask.sum() * 625 / 1e6, 1),
    "psr_pct": round(100 * psr_mask.mean(), 1),
    "phase1_ice_km2": round(phase1_ice.sum() * 625 / 1e6, 1),
    "ml_ice_50_km2": round(ml_ice_50.sum() * 625 / 1e6, 1),
    "ml_ice_70_km2": round(ml_ice_70.sum() * 625 / 1e6, 1),
    "ml_ice_90_km2": round(ml_ice_90.sum() * 625 / 1e6, 1),
    "roc_auc": 0.76,
    "model": "XGBoost (500 trees, lr=0.05)",
    "label_source": "NASA LRO Mini-RF S-band CPR",
    "cpr_median": round(float(np.nanmedian(cpr)), 4),
    "total_pixels": int(np.isfinite(cpr).sum()),
    "image_width": w,
    "image_height": h,
    "feature_importance": {
        "ODD": 0.449, "VOL": 0.201, "CPR": 0.095,
        "HLX": 0.070, "TRT": 0.070, "EVN": 0.061, "SRD": 0.053
    },
    "craters": [
        {"name": "Faustini", "lat": -87.3, "lon": 77.0},
        {"name": "Shoemaker", "lat": -88.1, "lon": 44.9},
        {"name": "Haworth", "lat": -87.5, "lon": -5.0},
        {"name": "Cabeus", "lat": -85.3, "lon": -35.7},
        {"name": "Shackleton", "lat": -89.7, "lon": 129.8}
    ]
}
data_dir = Path(r"D:/1_p3-isro/site/data")
data_dir.mkdir(parents=True, exist_ok=True)
with open(data_dir / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nDone in {time.time()-t0:.0f}s")
for p in sorted(OUT.glob("*.png")):
    sz = round(p.stat().st_size / 1e6, 2)
    print(f"  {p.name:25s} {sz:6.2f} MB")
