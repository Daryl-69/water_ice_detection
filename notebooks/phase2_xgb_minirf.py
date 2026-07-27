"""
Phase 2 v2 — Non-Circular ML Training with XGBoost
==================================================
Labels:  Mini-RF S-band CPR (NASA LRO, independent sensor)
Features: All 7 DFSAR L-band parameters (ISRO Chandrayaan-2)
Model:   XGBoost (gradient boosted trees)

This eliminates the circular label problem:
- Labels come from S-band radar (2.38 GHz, NASA)
- Features come from L-band radar (1.25 GHz, ISRO)
- Different spacecraft, different frequency = genuine cross-sensor learning
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import joblib, warnings, time, struct
warnings.filterwarnings("ignore")
from pathlib import Path

t0 = time.time()

# ── Paths ─────────────────────────────────────────────────────────────────────
DFSAR  = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA   = Path(r"D:/1_p3-isro/datasets/lola")
MINIRF = Path(r"D:/1_p3-isro/datasets/minirf/global_cpr_128ppd_simp_0c.img")
OUT    = Path(r"D:/1_p3-isro/notebooks")

BAND_FILES = {
    "CPR": DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif",
    "VOL": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_vol_xx_fp_xx_xxx.tif",
    "EVN": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_evn_xx_fp_xx_xxx.tif",
    "ODD": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_odd_xx_fp_xx_xxx.tif",
    "HLX": DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_hlx_xx_fp_xx_xxx.tif",
    "TRT": DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_trt_xx_fp_xx_xxx.tif",
    "SRD": DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_srd_xx_fp_xx_xxx.tif",
}
BAND_NAMES = list(BAND_FILES.keys())

PSR_F = LOLA / "LPSR_85S_060M_201608.JP2"
DEM_F = LOLA / "LM7_final_adj_5mpp_surf.tif"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Mini-RF CPR (PDS3 binary) & crop to south pole
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Mini-RF CPR (4 GB PDS3 binary)")
print("=" * 60)

# From the PDS label:
# LINES = 23040, LINE_SAMPLES = 46080, SAMPLE_BITS = 32 (PC_REAL = little-endian float32)
# MAP_RESOLUTION = 128 pix/deg, Simple Cylindrical, -90 to 90 lat, -180 to 180 lon
MRF_LINES = 23040
MRF_SAMPLES = 46080
MRF_RES = 128  # pixels per degree
MRF_MISSING = -1.7976931e+308

# We only need south pole: lat < -75° (bottom portion of the image)
# Row for latitude: row = (90 - lat) * resolution
# lat = -75 → row = (90 - (-75)) * 128 = 165 * 128 = 21120
# lat = -90 → row = (90 - (-90)) * 128 = 180 * 128 = 23040
START_ROW = 21120  # lat = -75
END_ROW = 23040    # lat = -90

n_rows = END_ROW - START_ROW  # 1920 rows
n_cols = MRF_SAMPLES           # 46080 columns

print(f"  Cropping to south pole: rows {START_ROW}-{END_ROW} ({n_rows} rows × {n_cols} cols)")

# Read only south pole portion from binary file
byte_offset = START_ROW * n_cols * 4  # 4 bytes per float32
mrf_data = np.zeros((n_rows, n_cols), dtype=np.float32)

with open(MINIRF, 'rb') as f:
    f.seek(byte_offset)
    raw = f.read(n_rows * n_cols * 4)
    mrf_data = np.frombuffer(raw, dtype='<f4').reshape(n_rows, n_cols).copy()

# Replace missing values with NaN
mrf_data[mrf_data < -1e+300] = np.nan
mrf_data[mrf_data <= 0] = np.nan

valid_mrf = np.isfinite(mrf_data)
print(f"  Mini-RF south pole: {n_rows} × {n_cols} pixels")
print(f"  Valid pixels: {valid_mrf.sum():,} ({100*valid_mrf.mean():.1f}%)")
print(f"  CPR range: [{np.nanmin(mrf_data):.3f}, {np.nanmax(mrf_data):.3f}]")
print(f"  CPR median: {np.nanmedian(mrf_data):.3f}")

# Create georeferencing for Mini-RF crop (Simple Cylindrical)
# Mini-RF uses planetocentric coordinates, simple cylindrical projection
# Pixel (0,0) of full image = lon=-180, lat=+90
# Pixel (col, row) → lon = -180 + col/128, lat = 90 - row/128
# For our crop starting at row 21120:
# Top-left: lon=-180, lat=90 - 21120/128 = 90 - 165 = -75
# Bottom-right: lon=+180, lat=90 - 23040/128 = 90 - 180 = -90
from rasterio.crs import CRS

# Save as temporary GeoTIFF for reprojection
mrf_crs = CRS.from_proj4("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs")
# For simple cylindrical: transform from pixel to CRS coordinates
# At equator: 1 degree = 2*pi*R/360 = 2*pi*1737400/360 = 30,334.69 m
# pixel size = 30334.69 / 128 = 236.99 m at equator
moon_radius = 1737400.0  # meters
deg_to_m = np.pi * moon_radius / 180.0  # meters per degree at equator
pixel_size_m = deg_to_m / MRF_RES

# Bounds in meters (equirectangular)
west_m = -180 * deg_to_m
east_m = 180 * deg_to_m
north_m = -75 * deg_to_m  # lat -75
south_m = -90 * deg_to_m  # lat -90

mrf_transform = from_bounds(west_m, south_m, east_m, north_m, n_cols, n_rows)

mrf_tif = OUT / "minirf_cpr_southpole.tif"
with rasterio.open(
    mrf_tif, "w", driver="GTiff",
    height=n_rows, width=n_cols, count=1, dtype="float32",
    crs=mrf_crs, transform=mrf_transform, nodata=np.nan
) as dst:
    dst.write(mrf_data, 1)
print(f"  Saved temp GeoTIFF: {mrf_tif}")
print(f"  Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load DFSAR bands + reproject Mini-RF to DFSAR grid
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading 7 DFSAR bands + reprojecting Mini-RF")
print("=" * 60)

bands = {}
ref_crs = ref_tf = None
for name, path in BAND_FILES.items():
    print(f"  Loading {name}...", end=" ", flush=True)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if ref_crs is None:
            ref_crs = src.crs
            ref_tf = src.transform
            H, W = data.shape
    data[data <= 0] = np.nan
    if name in ("VOL", "EVN", "ODD", "HLX"):
        data = np.log10(data * 1e12)
        data[~np.isfinite(data)] = np.nan
    bands[name] = data
    print(f"[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")

print(f"\n  DFSAR grid: {H} × {W}")

# Reproject Mini-RF to DFSAR grid
print("  Reprojecting Mini-RF CPR to DFSAR grid...")
mrf_on_dfsar = np.full((H, W), np.nan, dtype=np.float32)
with rasterio.open(mrf_tif) as src:
    reproject(
        source=rasterio.band(src, 1),
        destination=mrf_on_dfsar,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_tf,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

mrf_valid = np.isfinite(mrf_on_dfsar)
print(f"  Mini-RF on DFSAR grid: {mrf_valid.sum():,} valid pixels ({100*mrf_valid.mean():.1f}%)")
print(f"  Mini-RF CPR range on grid: [{np.nanmin(mrf_on_dfsar):.3f}, {np.nanmax(mrf_on_dfsar):.3f}]")
print(f"  Time: {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load PSR + slope
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Loading PSR + DEM")

with rasterio.open(PSR_F) as src:
    psr_raw = src.read(1).astype(np.float32)
    psr_crs, psr_tf = src.crs, src.transform
psr_binary = ((psr_raw * 0.000025 + 0.5) > 0.5).astype(np.float32)
psr_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(psr_binary, psr_on_cpr,
          src_transform=psr_tf, src_crs=psr_crs,
          dst_transform=ref_tf, dst_crs=ref_crs,
          resampling=Resampling.nearest)
psr_mask = (psr_on_cpr > 0.5).astype(np.uint8)

with rasterio.open(DEM_F) as src:
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
print(f"  PSR: {100*psr_mask.mean():.1f}% | DEM+slope computed")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Create INDEPENDENT labels from Mini-RF
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Creating independent labels from Mini-RF")
print("=" * 60)

# Valid: pixel has data in ALL 7 DFSAR bands AND Mini-RF
valid_all = np.ones((H, W), dtype=bool)
for name in BAND_NAMES:
    valid_all &= np.isfinite(bands[name])

# Pixels with BOTH DFSAR and Mini-RF coverage
valid_both = valid_all & mrf_valid
print(f"  Pixels with DFSAR data: {valid_all.sum():,}")
print(f"  Pixels with Mini-RF data: {mrf_valid.sum():,}")
print(f"  Pixels with BOTH: {valid_both.sum():,}")

# LABELS from Mini-RF (S-band, independent sensor):
# ICE: Mini-RF CPR > 1.0 inside PSR (S-band anomaly in permanent shadow)
# NOT ICE: Mini-RF CPR < 0.5 in sunlit areas OR Mini-RF CPR < 0.3 anywhere
slope_ok = (slope < 25.0) | ~np.isfinite(slope)

ice_label = valid_both & (mrf_on_dfsar > 1.0) & (psr_mask == 1) & slope_ok
not_ice_label = valid_both & (psr_mask == 0) & (mrf_on_dfsar < 0.5)

n_ice = ice_label.sum()
n_not_ice = not_ice_label.sum()
print(f"\n  ICE labels (Mini-RF CPR>1.0, in PSR):    {n_ice:,}")
print(f"  NOT ICE labels (sunlit, Mini-RF CPR<0.5): {n_not_ice:,}")

if n_ice == 0:
    print("\n  WARNING: No ice labels found! Trying lower threshold...")
    for thresh in [0.8, 0.6, 0.5, 0.3]:
        ice_label = valid_both & (mrf_on_dfsar > thresh) & (psr_mask == 1) & slope_ok
        n_ice = ice_label.sum()
        print(f"    Mini-RF CPR > {thresh}: {n_ice:,} ice pixels")
        if n_ice > 100:
            print(f"    Using threshold {thresh}")
            break

# Subsample NOT ICE for balance
rng = np.random.RandomState(42)
not_ice_flat = np.where(not_ice_label.ravel())[0]
ice_flat = np.where(ice_label.ravel())[0]

n_sample_neg = min(100_000, n_not_ice)
not_ice_sampled = rng.choice(not_ice_flat, size=n_sample_neg, replace=False)

all_indices = np.concatenate([not_ice_sampled, ice_flat])
all_labels = np.concatenate([np.zeros(len(not_ice_sampled)), np.ones(len(ice_flat))])

# Feature stack for all valid pixels
feature_stack = np.column_stack([bands[name][valid_all] for name in BAND_NAMES])
n_valid = valid_all.sum()

# Map flat indices to feature rows
flat_to_row = np.full(H * W, -1, dtype=np.int64)
flat_to_row[valid_all.ravel()] = np.arange(n_valid)

train_rows = flat_to_row[all_indices]
mask_valid_rows = train_rows >= 0
train_rows = train_rows[mask_valid_rows]
all_labels = all_labels[mask_valid_rows]

X_all = feature_stack[train_rows]
y_all = all_labels

print(f"\n  Training set: {len(X_all):,} samples")
print(f"    ICE: {int(y_all.sum()):,}")
print(f"    NOT ICE: {int((y_all == 0).sum()):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Train XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training XGBoost (gradient boosted trees)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# Calculate class weight
n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())
scale_pos = n_neg / max(n_pos, 1)
print(f"  Class balance: scale_pos_weight = {scale_pos:.1f}")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

t1 = time.time()
model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
print(f"  Training done in {time.time()-t1:.1f}s")

# Evaluate
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["NOT ICE", "ICE"]))

# Feature importance
importances = model.feature_importances_
print("  Feature Importance (from XGBoost):")
for name, imp in sorted(zip(BAND_NAMES, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"    {name:4s}: {imp:.3f}  {bar}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\n  ROC AUC: {roc_auc:.4f}")

# Save model
joblib.dump({
    "model": model, "scaler": scaler, "bands": BAND_NAMES,
    "type": "xgboost", "label_source": "Mini-RF S-band CPR",
    "roc_auc": roc_auc
}, OUT / "xgb_model.joblib")
print(f"  Model saved: {OUT / 'xgb_model.joblib'}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Predict on full DFSAR mosaic
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Predicting on all valid pixels")
print("=" * 60)

feature_stack_s = scaler.transform(feature_stack)

CHUNK = 500_000
n_chunks = (n_valid + CHUNK - 1) // CHUNK
proba_valid = np.zeros(n_valid, dtype=np.float32)

t2 = time.time()
for i in range(n_chunks):
    start = i * CHUNK
    end = min(start + CHUNK, n_valid)
    proba_valid[start:end] = model.predict_proba(feature_stack_s[start:end])[:, 1].astype(np.float32)
    if (i + 1) % 20 == 0 or i == n_chunks - 1:
        print(f"  Chunk {i+1}/{n_chunks}")

# Map to full grid
prob_map = np.full((H, W), np.nan, dtype=np.float32)
prob_map[valid_all] = proba_valid

# Apply PSR constraint (ice impossible in sunlight)
prob_map_psr = prob_map.copy()
prob_map_psr[psr_mask == 0] = 0.0

print(f"  Prediction done in {time.time()-t2:.0f}s")

for thresh in [0.3, 0.5, 0.7, 0.9]:
    n_px = np.nansum(prob_map_psr > thresh)
    area = n_px * 625 / 1e6
    print(f"  P(ice) > {thresh} (PSR only): {n_px:,} px = {area:.1f} km2")

# Save GeoTIFFs
with rasterio.open(
    OUT / "ice_probability_xgb.tif", "w", driver="GTiff",
    height=H, width=W, count=1, dtype="float32",
    crs=ref_crs, transform=ref_tf, nodata=np.nan
) as dst:
    dst.write(prob_map_psr, 1)
print(f"  Saved: ice_probability_xgb.tif")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Phase 1 reconstruction for comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Phase 1 comparison")

VOL_75 = float(np.nanpercentile(bands["VOL"][np.isfinite(bands["VOL"])], 75))
slope_ok_p1 = (slope < 20.0) | ~np.isfinite(slope)
phase1_ice = valid_all & (bands["CPR"] > 1.0) & (bands["VOL"] > VOL_75) & (psr_mask == 1) & slope_ok_p1
phase1_area = phase1_ice.sum() * 625 / 1e6

ml_ice = prob_map_psr > 0.5
ml_area = ml_ice.sum() * 625 / 1e6

ml_only = ml_ice & ~phase1_ice
p1_only = phase1_ice & ~ml_ice
both = ml_ice & phase1_ice

print(f"  Phase 1 (thresholds):  {phase1_area:.1f} km2")
print(f"  Phase 2 (XGBoost):     {ml_area:.1f} km2")
print(f"  Both agree:            {both.sum()*625/1e6:.1f} km2")
print(f"  XGBoost only (new):    {ml_only.sum()*625/1e6:.1f} km2")
print(f"  Threshold only:        {p1_only.sum()*625/1e6:.1f} km2")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Generate all plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Generating plots")

def ds(arr, f=4):
    return arr[::f, ::f]

# --- Plot 1: Corrected ML vs Threshold comparison ---
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
fig.suptitle("XGBoost ML (Mini-RF Labels) vs Thresholds — PSR-Constrained\n"
             "Labels: NASA LRO Mini-RF S-band | Features: ISRO DFSAR L-band",
             fontsize=14, fontweight="bold")

cpr = bands["CPR"]

ax = axes[0, 0]
ax.imshow(ds(np.clip(cpr, 0, 3)), cmap="inferno", vmin=0, vmax=3, aspect="auto")
p1_show = np.zeros((H, W), dtype=np.uint8); p1_show[phase1_ice] = 255
ax.imshow(np.ma.masked_where(ds(p1_show)==0, ds(p1_show)),
          cmap="cool", vmin=0, vmax=255, alpha=0.85, aspect="auto")
ax.set_title(f"Phase 1: Rule-Based Thresholds\n{phase1_area:.1f} km²", fontsize=12)

ax = axes[0, 1]
cmap_prob = plt.cm.RdYlBu_r.copy(); cmap_prob.set_bad("#0d0d1a")
im = ax.imshow(ds(prob_map_psr), cmap=cmap_prob, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="P(ice)", shrink=0.8)
try: ax.contour(ds(psr_mask), levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.5)
except: pass
ax.set_title(f"Phase 2: XGBoost ML (PSR only)\n{ml_area:.1f} km² at P > 0.5", fontsize=12)

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
ax.set_title("Agreement Map", fontsize=12)

ax = axes[1, 1]
ax.imshow(ds(np.clip(cpr, 0, 3)), cmap="inferno", vmin=0, vmax=3, aspect="auto")
hc = np.zeros((H, W), dtype=np.uint8); hc[prob_map_psr > 0.7] = 255
hc_area = (prob_map_psr > 0.7).sum() * 625 / 1e6
ax.imshow(np.ma.masked_where(ds(hc)==0, ds(hc)),
          cmap="winter", vmin=0, vmax=255, alpha=0.85, aspect="auto")
ax.set_title(f"High Confidence (P > 0.7)\n{hc_area:.1f} km²", fontsize=12)

plt.tight_layout()
plt.savefig(OUT / "08_xgb_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_xgb_vs_threshold.png")

# --- Plot 2: Feature importance + ROC ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost Model Analysis (Mini-RF Independent Labels)", fontsize=13, fontweight="bold")

sorted_idx = np.argsort(importances)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, 7))
ax1.barh([BAND_NAMES[i] for i in sorted_idx], importances[sorted_idx],
         color=colors, edgecolor="white", linewidth=0.5)
ax1.set_xlabel("Feature Importance")
ax1.set_title("Which DFSAR Bands Predict Mini-RF Ice?")
for i, idx in enumerate(sorted_idx):
    ax1.text(importances[idx] + 0.005, i, f"{importances[idx]:.3f}", va="center", fontsize=10)

ax2.plot(fpr, tpr, color="#00d4ff", lw=2.5, label=f"XGBoost (AUC = {roc_auc:.3f})")
ax2.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
ax2.fill_between(fpr, tpr, alpha=0.1, color="#00d4ff")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve (Cross-Sensor Validation)")
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUT / "09_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 09_xgb_feature_importance.png")

# --- Plot 3: Confusion matrix ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost Evaluation (Independent Labels)", fontsize=13, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
im = ax1.imshow(cm, cmap="Blues")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["NOT ICE", "ICE"])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["NOT ICE", "ICE"])
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual (Mini-RF)")
ax1.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i,j] > cm.max()/2 else "black")

# Mini-RF vs DFSAR CPR scatter
sample_both = valid_both.ravel()
sample_idx = np.where(sample_both)[0]
if len(sample_idx) > 10000:
    sample_idx = rng.choice(sample_idx, size=10000, replace=False)
s_rows = np.unravel_index(sample_idx, (H, W))
dfsar_cpr_sample = bands["CPR"][s_rows]
minirf_cpr_sample = mrf_on_dfsar[s_rows]

ax2.scatter(dfsar_cpr_sample, minirf_cpr_sample, s=2, alpha=0.3, c='#00d4ff', edgecolors='none')
ax2.set_xlabel("DFSAR L-band CPR (ISRO)")
ax2.set_ylabel("Mini-RF S-band CPR (NASA)")
ax2.set_title("Cross-Sensor CPR Comparison")
ax2.axhline(1.0, color='red', ls='--', alpha=0.5, label='Ice threshold')
ax2.axvline(1.0, color='red', ls='--', alpha=0.5)
ax2.set_xlim(0, 3); ax2.set_ylim(0, 3)
ax2.legend()

plt.tight_layout()
plt.savefig(OUT / "10_xgb_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 10_xgb_evaluation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - t0
print("\n" + "=" * 60)
print("  PHASE 2 v2 COMPLETE — XGBoost + Mini-RF Labels")
print("=" * 60)
print(f"  Model:    XGBoost (500 trees, lr=0.05)")
print(f"  Labels:   Mini-RF S-band CPR (NASA LRO)")
print(f"  Features: {', '.join(BAND_NAMES)} (ISRO DFSAR)")
print(f"  Training: {len(X_train):,} samples | Test: {len(X_test):,}")
print(f"  ROC AUC:  {roc_auc:.4f}")
print(f"")
print(f"  Phase 1 (thresholds):  {phase1_area:.1f} km²")
print(f"  Phase 2 (XGBoost):     {ml_area:.1f} km²")
print(f"  High confidence P>0.7: {hc_area:.1f} km²")
print(f"  Both methods agree:    {both.sum()*625/1e6:.1f} km²")
print(f"")
print(f"  Feature importance:")
for name, imp in sorted(zip(BAND_NAMES, importances), key=lambda x: -x[1]):
    print(f"    {name:4s}: {imp:.1%}")
print(f"")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print("=" * 60)
