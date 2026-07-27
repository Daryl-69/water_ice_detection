"""
Generate enhanced 3D assets: normal map + data texture for click-to-inspect.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import json, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

t0 = time.time()
DFSAR = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA  = Path(r"D:/1_p3-isro/datasets/lola")
OUT   = Path(r"D:/1_p3-isro/site/assets/3d")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 1024

# ── Load everything ──
print("Loading all data...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif") as src:
    cpr_full = src.read(1).astype(np.float32)
    ref_crs, ref_tf = src.crs, src.transform
    H, W = cpr_full.shape
cpr_full[cpr_full <= 0] = np.nan

with rasterio.open(LOLA / "LM7_final_adj_5mpp_surf.tif") as src:
    dem_raw = src.read(1).astype(np.float32)
    dem_crs, dem_tf = src.crs, src.transform
dem_raw[dem_raw < -9000] = np.nan
dem_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(dem_raw, dem_on_cpr, src_transform=dem_tf, src_crs=dem_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.bilinear)
dem_on_cpr[dem_on_cpr == 0] = np.nan

with rasterio.open(LOLA / "LPSR_85S_060M_201608.JP2") as src:
    psr_raw = src.read(1).astype(np.float32)
    psr_crs, psr_tf = src.crs, src.transform
psr_binary = ((psr_raw * 0.000025 + 0.5) > 0.5).astype(np.float32)
psr_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(psr_binary, psr_on_cpr, src_transform=psr_tf, src_crs=psr_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.nearest)
psr_mask = (psr_on_cpr > 0.5).astype(np.uint8)

with rasterio.open(Path(r"D:/1_p3-isro/notebooks/ice_probability_xgb.tif")) as src:
    prob_map = src.read(1)

dy, dx = np.gradient(np.where(np.isfinite(dem_on_cpr), dem_on_cpr, 0.0), 25.0)
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope[~np.isfinite(dem_on_cpr)] = np.nan

# ── Downsample ──
print(f"Downsampling to {SIZE}x{SIZE}...")
def ds(arr, size=SIZE):
    img = Image.fromarray(np.where(np.isfinite(arr), arr, 0).astype(np.float32))
    return np.array(img.resize((size, size), Image.BILINEAR))

cpr_s = ds(np.clip(cpr_full, 0, 3))
dem_s = ds(dem_on_cpr)
prob_s = ds(prob_map)
psr_s = ds(psr_mask.astype(np.float32))
slope_s = ds(slope)

dem_min = float(np.nanmin(dem_s[dem_s != 0]))
dem_max = float(np.nanmax(dem_s))
dem_s[dem_s == 0] = dem_min
dem_s[~np.isfinite(dem_s)] = dem_min
dem_norm = np.clip((dem_s - dem_min) / (dem_max - dem_min), 0, 1)

# ═══ 1. Heightmap (8-bit) ═══
print("Generating heightmap...")
Image.fromarray((dem_norm * 255).astype(np.uint8), mode='L').save(OUT / "heightmap_8bit.png")

# ═══ 2. Normal map from heightmap ═══
print("Generating normal map...")
strength = 2.0
dz_dx = np.gradient(dem_norm, axis=1) * strength
dz_dy = np.gradient(dem_norm, axis=0) * strength
normals = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
normals[:, :, 0] = -dz_dx
normals[:, :, 1] = -dz_dy
normals[:, :, 2] = 1.0
length = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
normals /= length
normal_rgb = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
Image.fromarray(normal_rgb).save(OUT / "normalmap.png")

# ═══ 3. Realistic lunar surface texture ═══
print("Generating realistic surface texture...")
rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
rgba[:, :, 3] = 255

# Base gray from elevation-based shading (simulating regolith albedo)
# Higher = slightly lighter, crater floors = darker
elev_shade = np.clip(dem_norm * 0.4 + 0.15, 0.15, 0.55)

# Add surface roughness from slope (steeper = slightly different shade)
slope_factor = np.clip(slope_s / 40.0, 0, 1)
micro_rough = np.random.RandomState(42).uniform(0.9, 1.1, (SIZE, SIZE)).astype(np.float32)

# Lunar regolith is gray with subtle warm/cool shifts
base_r = np.clip(elev_shade * 155 * micro_rough + slope_factor * 10, 0, 200).astype(np.uint8)
base_g = np.clip(elev_shade * 150 * micro_rough + slope_factor * 5, 0, 195).astype(np.uint8)
base_b = np.clip(elev_shade * 148 * micro_rough - slope_factor * 5, 0, 190).astype(np.uint8)

rgba[:, :, 0] = base_r
rgba[:, :, 1] = base_g
rgba[:, :, 2] = base_b

# PSR regions are VERY dark (no sunlight ever reaches here)
psr_m = psr_s > 0.3
rgba[psr_m, 0] = np.clip(rgba[psr_m, 0].astype(int) * 0.15 + 5, 5, 30).astype(np.uint8)
rgba[psr_m, 1] = np.clip(rgba[psr_m, 1].astype(int) * 0.15 + 5, 5, 30).astype(np.uint8)
rgba[psr_m, 2] = np.clip(rgba[psr_m, 2].astype(int) * 0.18 + 8, 8, 40).astype(np.uint8)

# Ice deposits: subtle frost-blue tint (not glowing — realistic)
ice_mask = prob_s > 0.3
if ice_mask.any():
    p = prob_s[ice_mask].clip(0, 1)
    # Frost makes the surface slightly brighter and bluish
    rgba[ice_mask, 0] = np.clip(rgba[ice_mask, 0].astype(int) + (p * 25).astype(int), 0, 80).astype(np.uint8)
    rgba[ice_mask, 1] = np.clip(rgba[ice_mask, 1].astype(int) + (p * 40).astype(int), 0, 90).astype(np.uint8)
    rgba[ice_mask, 2] = np.clip(rgba[ice_mask, 2].astype(int) + (p * 65).astype(int), 0, 120).astype(np.uint8)

Image.fromarray(rgba).save(OUT / "texture_realistic.png")

# ═══ 4. Sci-fi ice glow texture (toggle-able) ═══
print("Generating sci-fi ice overlay...")
rgba2 = rgba.copy()
if ice_mask.any():
    p = prob_s[ice_mask].clip(0, 1)
    rgba2[ice_mask, 0] = np.clip((p * 60).astype(int), 0, 255).astype(np.uint8)
    rgba2[ice_mask, 1] = np.clip((p * 200 + 20).astype(int), 0, 255).astype(np.uint8)
    rgba2[ice_mask, 2] = np.clip((p * 255 + 40).astype(int), 0, 255).astype(np.uint8)
Image.fromarray(rgba2).save(OUT / "texture_ice_glow.png")

# ═══ 5. Data texture (RGBA encodes CPR, Ice%, Slope, PSR) ═══
print("Generating data texture...")
data_rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
# R = CPR (0-3 mapped to 0-255)
data_rgba[:, :, 0] = np.clip(cpr_s / 3.0 * 255, 0, 255).astype(np.uint8)
# G = Ice probability (0-1 mapped to 0-255)
data_rgba[:, :, 1] = np.clip(prob_s * 255, 0, 255).astype(np.uint8)
# B = Slope (0-45 mapped to 0-255)
data_rgba[:, :, 2] = np.clip(slope_s / 45.0 * 255, 0, 255).astype(np.uint8)
# A = PSR (0 or 255)
data_rgba[:, :, 3] = (psr_s > 0.3).astype(np.uint8) * 255
Image.fromarray(data_rgba).save(OUT / "data_texture.png")

# ═══ 6. CPR texture ═══
print("Generating CPR radar texture...")
cpr_norm = np.clip(cpr_s / 3.0, 0, 1)
cmap = plt.cm.inferno
rgba_cpr = (cmap(cpr_norm) * 255).astype(np.uint8)
rgba_cpr[cpr_s <= 0.001] = [15, 12, 30, 255]
Image.fromarray(rgba_cpr).save(OUT / "texture_cpr.png")

# ═══ 7. Updated metadata ═══
meta = {
    "dem_min_m": dem_min,
    "dem_max_m": dem_max,
    "dem_range_m": dem_max - dem_min,
    "grid_size": SIZE,
    "pixel_size_m": round(25 * (H / SIZE), 1),
    "total_width_km": round(25 * W / 1000, 1),
    "total_height_km": round(25 * H / 1000, 1),
    "craters": [
        {"name": "Faustini", "px": int(0.69*SIZE), "py": int(0.55*SIZE), "radius_km": 39},
        {"name": "Shoemaker", "px": int(0.55*SIZE), "py": int(0.65*SIZE), "radius_km": 51},
        {"name": "Haworth", "px": int(0.42*SIZE), "py": int(0.50*SIZE), "radius_km": 28},
        {"name": "Shackleton", "px": int(0.53*SIZE), "py": int(0.32*SIZE), "radius_km": 11},
    ]
}
with open(OUT / "terrain_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone in {time.time()-t0:.0f}s")
for p in sorted(OUT.glob("*")):
    print(f"  {p.name:30s} {p.stat().st_size/1e6:.2f} MB")
