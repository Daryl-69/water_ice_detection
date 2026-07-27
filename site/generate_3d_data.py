"""
Generate NASA-quality south pole visualization textures.
- Photorealistic hillshade from LOLA DEM (looks like real lunar photography)
- Semi-transparent blue ice overlay (like NASA LEND imagery)
- High resolution 2048x2048 for sharp detail
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

t0 = time.time()
DFSAR = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA  = Path(r"D:/1_p3-isro/datasets/lola")
OUT   = Path(r"D:/1_p3-isro/site/assets/3d")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 2048  # Higher res for sharp detail

# ── Load data ──
print("Loading data...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif") as src:
    cpr_full = src.read(1).astype(np.float32)
    ref_crs, ref_tf = src.crs, src.transform
    H, W = cpr_full.shape
cpr_full[cpr_full <= 0] = np.nan

with rasterio.open(LOLA / "LM7_final_adj_5mpp_surf.tif") as src:
    dem_raw = src.read(1).astype(np.float32)
    dem_crs, dem_tf = src.crs, src.transform
dem_raw[dem_raw < -9000] = np.nan
dem = np.zeros((H, W), dtype=np.float32)
reproject(dem_raw, dem, src_transform=dem_tf, src_crs=dem_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.bilinear)
dem[dem == 0] = np.nan

with rasterio.open(LOLA / "LPSR_85S_060M_201608.JP2") as src:
    psr_raw = src.read(1).astype(np.float32)
    psr_crs, psr_tf = src.crs, src.transform
psr_binary = ((psr_raw * 0.000025 + 0.5) > 0.5).astype(np.float32)
psr = np.zeros((H, W), dtype=np.float32)
reproject(psr_binary, psr, src_transform=psr_tf, src_crs=psr_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.nearest)
psr_mask = (psr > 0.5).astype(np.uint8)

with rasterio.open(Path(r"D:/1_p3-isro/notebooks/ice_probability_xgb.tif")) as src:
    prob = src.read(1)

# Slope
dy, dx = np.gradient(np.where(np.isfinite(dem), dem, 0.0), 25.0)
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope[~np.isfinite(dem)] = np.nan

# ── Downsample to SIZE ──
print(f"Downsampling to {SIZE}x{SIZE}...")
def ds(arr, sz=SIZE):
    return np.array(Image.fromarray(np.where(np.isfinite(arr), arr, 0).astype(np.float32)).resize((sz, sz), Image.BILINEAR))

dem_s = ds(dem)
cpr_s = ds(np.clip(cpr_full, 0, 3))
prob_s = ds(prob)
psr_s = ds(psr_mask.astype(np.float32))
slope_s = ds(slope)

dem_min = float(np.nanmin(dem_s[dem_s != 0]))
dem_max = float(np.nanmax(dem_s))
dem_s[dem_s == 0] = dem_min
dem_s[~np.isfinite(dem_s)] = dem_min
dem_norm = np.clip((dem_s - dem_min) / (dem_max - dem_min), 0, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# HILLSHADE — Photorealistic lunar surface
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing hillshade (NASA-quality)...")

# Light direction: low angle from left (azimuth 315°, altitude 15°)
# This creates dramatic crater shadows like in real lunar imagery
az = 315 * np.pi / 180   # azimuth
alt = 15 * np.pi / 180    # altitude (low sun = harsh shadows)

# Surface normals from DEM gradient
cell_size = 25.0 * (H / SIZE)  # meters per pixel at this resolution
dz_dy, dz_dx = np.gradient(dem_s, cell_size)

# Hillshade formula (standard Lambertian)
slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
aspect = np.arctan2(-dz_dy, dz_dx)

hillshade = (np.sin(alt) * np.cos(slope_rad) +
             np.cos(alt) * np.sin(slope_rad) * np.cos(az - aspect))
hillshade = np.clip(hillshade, 0, 1)

# Multiple light sources for realism (soft fill light from opposite side)
az2 = 135 * np.pi / 180
alt2 = 25 * np.pi / 180
hs2 = (np.sin(alt2) * np.cos(slope_rad) +
       np.cos(alt2) * np.sin(slope_rad) * np.cos(az2 - aspect))
hs2 = np.clip(hs2, 0, 1)

# Blend: 70% main light, 30% fill
hs_blend = hillshade * 0.7 + hs2 * 0.3

# Apply ambient occlusion (deeper craters get darker)
# Approximate using elevation: lower = darker
ao = dem_norm * 0.3 + 0.7  # range 0.7 to 1.0

hs_final = hs_blend * ao

# PSR regions: much darker (permanently shadowed)
hs_final[psr_s > 0.3] *= 0.15

# ═══ 1. Realistic Surface Texture ═══
print("Generating photorealistic surface...")
rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
rgba[:, :, 3] = 255

# Lunar regolith: gray with very subtle warm tint
# Based on actual lunar albedo (~0.12 average)
gray = np.clip(hs_final * 200 + 15, 10, 220)
rgba[:, :, 0] = np.clip(gray * 1.01, 0, 220).astype(np.uint8)  # very slight warmth
rgba[:, :, 1] = gray.astype(np.uint8)
rgba[:, :, 2] = np.clip(gray * 0.97, 0, 215).astype(np.uint8)  # very slight cool

# Add subtle noise for regolith texture
rng = np.random.RandomState(42)
noise = rng.uniform(0.95, 1.05, (SIZE, SIZE)).astype(np.float32)
rgba[:, :, 0] = np.clip(rgba[:, :, 0].astype(float) * noise, 5, 225).astype(np.uint8)
rgba[:, :, 1] = np.clip(rgba[:, :, 1].astype(float) * noise, 5, 220).astype(np.uint8)
rgba[:, :, 2] = np.clip(rgba[:, :, 2].astype(float) * noise, 5, 218).astype(np.uint8)

Image.fromarray(rgba).save(OUT / "texture_realistic.png", optimize=True)
print(f"  Saved texture_realistic.png ({rgba.nbytes/1e6:.1f} MB raw)")

# ═══ 2. Surface + Ice Overlay (NASA LEND style — blue overlay) ═══
print("Generating ice overlay texture (NASA LEND style)...")
rgba_ice = rgba.copy()

# Blue ice overlay: semi-transparent blue where ice probability is high
ice_mask = prob_s > 0.15
if ice_mask.any():
    p = prob_s[ice_mask].clip(0, 1)
    # Blend blue into the surface
    alpha = (p ** 0.6) * 0.85  # ice visibility
    base_r = rgba_ice[ice_mask, 0].astype(float)
    base_g = rgba_ice[ice_mask, 1].astype(float)
    base_b = rgba_ice[ice_mask, 2].astype(float)
    
    # NASA LEND-style blue: dark blue for low, bright blue/white for high
    ice_r = p * 100 + 20
    ice_g = p * 140 + 40
    ice_b = p * 255 + 80
    
    rgba_ice[ice_mask, 0] = np.clip(base_r * (1-alpha) + ice_r * alpha, 0, 255).astype(np.uint8)
    rgba_ice[ice_mask, 1] = np.clip(base_g * (1-alpha) + ice_g * alpha, 0, 255).astype(np.uint8)
    rgba_ice[ice_mask, 2] = np.clip(base_b * (1-alpha) + ice_b * alpha, 0, 255).astype(np.uint8)

Image.fromarray(rgba_ice).save(OUT / "texture_ice_glow.png", optimize=True)

# ═══ 3. CPR Radar texture ═══
print("Generating CPR radar texture...")
import matplotlib.pyplot as plt
cpr_norm = np.clip(cpr_s / 3.0, 0, 1)
cmap = plt.cm.inferno
rgba_cpr = (cmap(cpr_norm) * 255).astype(np.uint8)
rgba_cpr[cpr_s <= 0.001] = [15, 12, 30, 255]
Image.fromarray(rgba_cpr).save(OUT / "texture_cpr.png", optimize=True)

# ═══ 4. Heightmap ═══
print("Saving heightmap...")
Image.fromarray((dem_norm * 255).astype(np.uint8), mode='L').save(OUT / "heightmap_8bit.png")

# ═══ 5. Normal map ═══
print("Generating normal map...")
strength = 2.5
nz_dx = np.gradient(dem_norm, axis=1) * strength
nz_dy = np.gradient(dem_norm, axis=0) * strength
normals = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
normals[:, :, 0] = -nz_dx
normals[:, :, 1] = -nz_dy
normals[:, :, 2] = 1.0
length = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
normals /= length
Image.fromarray(((normals * 0.5 + 0.5) * 255).astype(np.uint8)).save(OUT / "normalmap.png")

# ═══ 6. Data texture ═══
print("Generating data texture...")
data_rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
data_rgba[:, :, 0] = np.clip(cpr_s / 3.0 * 255, 0, 255).astype(np.uint8)
data_rgba[:, :, 1] = np.clip(prob_s * 255, 0, 255).astype(np.uint8)
data_rgba[:, :, 2] = np.clip(slope_s / 45.0 * 255, 0, 255).astype(np.uint8)
data_rgba[:, :, 3] = (psr_s > 0.3).astype(np.uint8) * 255
Image.fromarray(data_rgba).save(OUT / "data_texture.png")

# ═══ 7. Metadata with MORE craters ═══
import json

# Crater positions (approximate pixel coords in our polar stereographic grid)
# Center of image = south pole
# These are estimated from known crater coordinates
craters = [
    {"name": "Shackleton",  "px": int(0.530*SIZE), "py": int(0.325*SIZE), "radius_km": 11, "depth_km": 4.2},
    {"name": "Faustini",    "px": int(0.690*SIZE), "py": int(0.555*SIZE), "radius_km": 39, "depth_km": 2.5},
    {"name": "Shoemaker",   "px": int(0.555*SIZE), "py": int(0.655*SIZE), "radius_km": 51, "depth_km": 2.0},
    {"name": "Haworth",     "px": int(0.425*SIZE), "py": int(0.505*SIZE), "radius_km": 28, "depth_km": 2.8},
    {"name": "Cabeus",      "px": int(0.280*SIZE), "py": int(0.490*SIZE), "radius_km": 52, "depth_km": 1.5},
    {"name": "de Gerlache", "px": int(0.440*SIZE), "py": int(0.380*SIZE), "radius_km": 16, "depth_km": 3.0},
    {"name": "Sverdrup",    "px": int(0.380*SIZE), "py": int(0.600*SIZE), "radius_km": 33, "depth_km": 2.2},
    {"name": "Amundsen",    "px": int(0.720*SIZE), "py": int(0.420*SIZE), "radius_km": 53, "depth_km": 1.8},
    {"name": "Scott",       "px": int(0.710*SIZE), "py": int(0.290*SIZE), "radius_km": 53, "depth_km": 2.0},
    {"name": "Nobile",      "px": int(0.400*SIZE), "py": int(0.680*SIZE), "radius_km": 37, "depth_km": 1.9},
]

meta = {
    "dem_min_m": dem_min,
    "dem_max_m": dem_max,
    "dem_range_m": dem_max - dem_min,
    "grid_size": SIZE,
    "pixel_size_m": round(25 * (H / SIZE), 1),
    "total_width_km": round(25 * W / 1000, 1),
    "total_height_km": round(25 * H / 1000, 1),
    "craters": craters
}
with open(OUT / "terrain_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone in {time.time()-t0:.0f}s")
for p in sorted(OUT.glob("*")):
    print(f"  {p.name:30s} {p.stat().st_size/1e6:.2f} MB")
