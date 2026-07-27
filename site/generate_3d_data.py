"""
Generate 3D-ready terrain data: heightmap + textures for Three.js viewer.
Exports downsampled DEM as grayscale PNG and overlay textures.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image
import matplotlib.pyplot as plt
import warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

t0 = time.time()
DFSAR = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA  = Path(r"D:/1_p3-isro/datasets/lola")
OUT   = Path(r"D:/1_p3-isro/site/assets/3d")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 1024  # output texture size

# ── Load DFSAR CPR (reference grid) ──
print("Loading CPR...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif") as src:
    cpr_full = src.read(1).astype(np.float32)
    ref_crs, ref_tf = src.crs, src.transform
    H, W = cpr_full.shape
cpr_full[cpr_full <= 0] = np.nan

# ── Load VOL ──
print("Loading VOL...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630my4rspeast_d_vol_xx_fp_xx_xxx.tif") as src:
    vol = src.read(1).astype(np.float32)
vol[vol <= 0] = np.nan
vol_log = np.log10(vol * 1e12)
vol_log[~np.isfinite(vol_log)] = np.nan

# ── Load DEM ──
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

# ── Load XGBoost probability ──
print("Loading ice probability...")
with rasterio.open(Path(r"D:/1_p3-isro/notebooks/ice_probability_xgb.tif")) as src:
    prob_map = src.read(1)

# ── Slope ──
print("Computing slope...")
dy, dx = np.gradient(np.where(np.isfinite(dem_on_cpr), dem_on_cpr, 0.0), 25.0)
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope[~np.isfinite(dem_on_cpr)] = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# Downsample everything to SIZE x SIZE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Downsampling to {SIZE}x{SIZE}...")

def downsample(arr, size):
    """Downsample with area averaging, handling NaNs."""
    img = Image.fromarray(np.where(np.isfinite(arr), arr, 0).astype(np.float32))
    img_r = img.resize((size, size), Image.BILINEAR)
    return np.array(img_r)

cpr_s = downsample(np.clip(cpr_full, 0, 3), SIZE)
dem_s = downsample(dem_on_cpr, SIZE)
prob_s = downsample(prob_map, SIZE)
psr_s = downsample(psr_mask.astype(np.float32), SIZE)
slope_s = downsample(slope, SIZE)
vol_s = downsample(vol_log, SIZE)

# Fill DEM gaps with minimum elevation for smooth terrain
dem_min = np.nanmin(dem_s[dem_s != 0])
dem_max = np.nanmax(dem_s)
dem_s[dem_s == 0] = dem_min
dem_s[~np.isfinite(dem_s)] = dem_min

print(f"  DEM range: [{dem_min:.0f}, {dem_max:.0f}] meters")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Heightmap (16-bit grayscale for precision)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating heightmap...")
dem_norm = (dem_s - dem_min) / (dem_max - dem_min)
dem_norm = np.clip(dem_norm, 0, 1)

# Save as 16-bit PNG for maximum precision
dem_16 = (dem_norm * 65535).astype(np.uint16)
Image.fromarray(dem_16, mode='I;16').save(OUT / "heightmap.png")

# Also save 8-bit version (Three.js standard)
dem_8 = (dem_norm * 255).astype(np.uint8)
Image.fromarray(dem_8, mode='L').save(OUT / "heightmap_8bit.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CPR terrain texture
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating CPR texture...")
cpr_norm = np.clip(cpr_s / 3.0, 0, 1)
cmap = plt.cm.inferno
rgba = (cmap(cpr_norm) * 255).astype(np.uint8)
# Darken areas with no data slightly
no_data = cpr_s <= 0.001
rgba[no_data] = [15, 12, 30, 255]
Image.fromarray(rgba).save(OUT / "texture_cpr.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Ice probability texture (glowing cyan/plasma)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating ice glow texture...")
# Create a beautiful ice visualization
rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

# Base: dark terrain
base_brightness = np.clip(cpr_norm * 0.3, 0, 0.3)
rgba[:, :, 0] = (base_brightness * 40).astype(np.uint8)
rgba[:, :, 1] = (base_brightness * 35).astype(np.uint8)
rgba[:, :, 2] = (base_brightness * 60).astype(np.uint8)
rgba[:, :, 3] = 255

# PSR shadows (dark blue tint)
psr_m = psr_s > 0.3
rgba[psr_m, 0] = np.clip(rgba[psr_m, 0].astype(int) + 5, 0, 255).astype(np.uint8)
rgba[psr_m, 1] = np.clip(rgba[psr_m, 1].astype(int) + 15, 0, 255).astype(np.uint8)
rgba[psr_m, 2] = np.clip(rgba[psr_m, 2].astype(int) + 40, 0, 255).astype(np.uint8)

# Ice glow (cyan to white gradient based on probability)
ice_mask = prob_s > 0.2
if ice_mask.any():
    p = prob_s[ice_mask].clip(0, 1)
    # Cyan glow: low prob = dark cyan, high prob = bright white-cyan
    rgba[ice_mask, 0] = np.clip((p * 80 + 0).astype(int), 0, 255).astype(np.uint8)
    rgba[ice_mask, 1] = np.clip((p * 220 + 30).astype(int), 0, 255).astype(np.uint8)
    rgba[ice_mask, 2] = np.clip((p * 255 + 50).astype(int), 0, 255).astype(np.uint8)

Image.fromarray(rgba).save(OUT / "texture_ice.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Slope safety texture
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating slope texture...")
rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
rgba[:, :, 3] = 255

safe = slope_s < 15
caution = (slope_s >= 15) & (slope_s < 25)
danger = slope_s >= 25
flat = slope_s <= 0.01

rgba[safe] = [20, 60, 20, 255]       # dark green
rgba[caution] = [80, 60, 10, 255]    # dark orange
rgba[danger] = [80, 20, 15, 255]     # dark red
rgba[flat] = [15, 15, 30, 255]       # no data

# Blend with CPR for context
cpr_gray = np.clip(cpr_norm * 80, 0, 80).astype(np.uint8)
rgba[:, :, 0] = np.clip(rgba[:, :, 0].astype(int) + cpr_gray, 0, 255).astype(np.uint8)
rgba[:, :, 1] = np.clip(rgba[:, :, 1].astype(int) + cpr_gray, 0, 255).astype(np.uint8)
rgba[:, :, 2] = np.clip(rgba[:, :, 2].astype(int) + cpr_gray, 0, 255).astype(np.uint8)

Image.fromarray(rgba).save(OUT / "texture_slope.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Combined beauty texture (for default view)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating combined beauty texture...")
rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
rgba[:, :, 3] = 255

# Lunar surface gray base from CPR
gray = np.clip(cpr_norm * 120 + 20, 20, 140).astype(np.uint8)
rgba[:, :, 0] = gray
rgba[:, :, 1] = gray
rgba[:, :, 2] = (gray * 1.1).clip(0, 160).astype(np.uint8)

# PSR regions get blue-tinted darkness
rgba[psr_m, 0] = (rgba[psr_m, 0] * 0.3).astype(np.uint8)
rgba[psr_m, 1] = (rgba[psr_m, 1] * 0.35).astype(np.uint8)
rgba[psr_m, 2] = np.clip((rgba[psr_m, 2].astype(float) * 0.5 + 25), 0, 255).astype(np.uint8)

# Ice deposits glow cyan
if ice_mask.any():
    p = prob_s[ice_mask].clip(0, 1)
    intensity = p ** 0.7  # make it more visible
    rgba[ice_mask, 0] = np.clip((intensity * 50 + rgba[ice_mask, 0] * (1 - intensity * 0.7)).astype(int), 0, 255).astype(np.uint8)
    rgba[ice_mask, 1] = np.clip((intensity * 200 + rgba[ice_mask, 1] * (1 - intensity * 0.5)).astype(int), 0, 255).astype(np.uint8)
    rgba[ice_mask, 2] = np.clip((intensity * 255 + rgba[ice_mask, 2] * (1 - intensity * 0.3)).astype(int), 0, 255).astype(np.uint8)

Image.fromarray(rgba).save(OUT / "texture_combined.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Metadata JSON
# ═══════════════════════════════════════════════════════════════════════════════
import json
meta = {
    "dem_min_m": float(dem_min),
    "dem_max_m": float(dem_max),
    "dem_range_m": float(dem_max - dem_min),
    "grid_size": SIZE,
    "pixel_size_m": 25 * (H / SIZE),  # approx
    "total_width_km": round(25 * W / 1000, 1),
    "total_height_km": round(25 * H / 1000, 1),
    "craters": [
        {"name": "Faustini", "px": int(0.69*SIZE), "py": int(0.55*SIZE), "radius": int(0.08*SIZE)},
        {"name": "Shoemaker", "px": int(0.55*SIZE), "py": int(0.65*SIZE), "radius": int(0.10*SIZE)},
        {"name": "Haworth", "px": int(0.42*SIZE), "py": int(0.50*SIZE), "radius": int(0.07*SIZE)},
        {"name": "Shackleton", "px": int(0.53*SIZE), "py": int(0.32*SIZE), "radius": int(0.03*SIZE)},
    ]
}
with open(OUT / "terrain_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nAll 3D assets generated in {time.time()-t0:.0f}s")
for p in sorted(OUT.glob("*")):
    sz = round(p.stat().st_size / 1e6, 2)
    print(f"  {p.name:30s} {sz:6.2f} MB")
