"""
Generate 3D textures using REAL LROC WAC photography of the lunar south pole.
This creates a photorealistic surface texture from actual LRO camera imagery.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image
import json, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

t0 = time.time()
DFSAR = Path(r"D:/1_p3-isro/datasets/dfsr_mosaic_fusti/data/derived/20250630")
LOLA  = Path(r"D:/1_p3-isro/datasets/lola")
OUT   = Path(r"D:/1_p3-isro/site/assets/3d")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 2048

# ── Load DFSAR grid (reference) ──
print("Loading DFSAR reference grid...")
with rasterio.open(DFSAR / "ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif") as src:
    cpr_full = src.read(1).astype(np.float32)
    ref_crs, ref_tf = src.crs, src.transform
    H, W = cpr_full.shape
cpr_full[cpr_full <= 0] = np.nan

# ── Load LROC WAC South Pole photograph ──
print("Loading LROC WAC photograph (141 MB)...")
wac_path = LOLA / "WAC_SOUTH_POLE.TIF"
with rasterio.open(wac_path) as src:
    wac_raw = src.read(1).astype(np.float32)
    wac_crs, wac_tf = src.crs, src.transform
    wac_h, wac_w = wac_raw.shape
    print(f"  WAC image: {wac_w}x{wac_h} pixels")
    print(f"  WAC CRS: {wac_crs}")
    print(f"  WAC transform: {wac_tf}")

# Reproject WAC to DFSAR grid
print("Reprojecting WAC photograph to DFSAR grid...")
wac_on_cpr = np.zeros((H, W), dtype=np.float32)
reproject(wac_raw, wac_on_cpr,
          src_transform=wac_tf, src_crs=wac_crs,
          dst_transform=ref_tf, dst_crs=ref_crs,
          resampling=Resampling.bilinear)
print(f"  Reprojected. Range: [{wac_on_cpr.min():.0f}, {wac_on_cpr.max():.0f}]")

# ── Load DEM ──
print("Loading DEM...")
with rasterio.open(LOLA / "LM7_final_adj_5mpp_surf.tif") as src:
    dem_raw = src.read(1).astype(np.float32)
    dem_crs, dem_tf = src.crs, src.transform
dem_raw[dem_raw < -9000] = np.nan
dem = np.zeros((H, W), dtype=np.float32)
reproject(dem_raw, dem, src_transform=dem_tf, src_crs=dem_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.bilinear)
dem[dem == 0] = np.nan

# ── Load PSR ──
print("Loading PSR...")
with rasterio.open(LOLA / "LPSR_85S_060M_201608.JP2") as src:
    psr_raw = src.read(1).astype(np.float32)
    psr_crs, psr_tf = src.crs, src.transform
psr_binary = ((psr_raw * 0.000025 + 0.5) > 0.5).astype(np.float32)
psr = np.zeros((H, W), dtype=np.float32)
reproject(psr_binary, psr, src_transform=psr_tf, src_crs=psr_crs,
          dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.nearest)
psr_mask = (psr > 0.5).astype(np.uint8)

# ── Load ice probability ──
print("Loading ice probability...")
with rasterio.open(Path(r"D:/1_p3-isro/notebooks/ice_probability_xgb.tif")) as src:
    prob = src.read(1)

# ── Slope ──
dy, dx = np.gradient(np.where(np.isfinite(dem), dem, 0.0), 25.0)
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope[~np.isfinite(dem)] = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# Downsample to SIZE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Downsampling all to {SIZE}x{SIZE}...")
def ds(arr, sz=SIZE):
    return np.array(Image.fromarray(np.where(np.isfinite(arr), arr, 0).astype(np.float32)).resize((sz, sz), Image.BILINEAR))

wac_s = ds(wac_on_cpr)
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
# 1. PHOTOREALISTIC TEXTURE — Real LROC WAC photograph
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating photorealistic LROC texture...")

# Normalize WAC to 0-255 (it's a grayscale camera image)
wac_valid = wac_s[wac_s > 0]
if len(wac_valid) > 0:
    p2, p98 = np.percentile(wac_valid, [2, 98])
    wac_norm = np.clip((wac_s - p2) / (p98 - p2), 0, 1)
else:
    wac_norm = np.clip(wac_s / (wac_s.max() + 1e-10), 0, 1)

# Where WAC has no data, fall back to hillshade
has_wac = wac_s > 0

# Compute hillshade as fallback for gaps
az, alt = 315 * np.pi / 180, 15 * np.pi / 180
cell_size = 25.0 * (H / SIZE)
dz_dy, dz_dx = np.gradient(dem_s, cell_size)
slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
aspect = np.arctan2(-dz_dy, dz_dx)
hs = np.clip(np.sin(alt) * np.cos(slope_rad) + np.cos(alt) * np.sin(slope_rad) * np.cos(az - aspect), 0, 1)

# Blend: use WAC where available, hillshade elsewhere
surface = np.where(has_wac, wac_norm, hs * 0.6)

# Create RGB image (real lunar surface is gray)
rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
rgba[:, :, 3] = 255

gray = np.clip(surface * 230 + 10, 5, 240)
# Very subtle warm tint matching real lunar regolith
rgba[:, :, 0] = np.clip(gray * 1.005, 0, 242).astype(np.uint8)
rgba[:, :, 1] = gray.astype(np.uint8)
rgba[:, :, 2] = np.clip(gray * 0.99, 0, 238).astype(np.uint8)

Image.fromarray(rgba).save(OUT / "texture_realistic.png", optimize=True)
print(f"  ✓ Photorealistic texture saved (LROC WAC imagery)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LROC + ICE OVERLAY (NASA LEND style — blue ice on real surface)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating LROC + ice overlay (NASA LEND style)...")
rgba_ice = rgba.copy()

ice_mask = prob_s > 0.15
if ice_mask.any():
    p = prob_s[ice_mask].clip(0, 1)
    alpha = (p ** 0.5) * 0.8
    base_r = rgba_ice[ice_mask, 0].astype(float)
    base_g = rgba_ice[ice_mask, 1].astype(float)
    base_b = rgba_ice[ice_mask, 2].astype(float)
    
    # NASA-style blue: deep blue → bright cyan
    ice_r = p * 80 + 15
    ice_g = p * 130 + 30
    ice_b = p * 255 + 60
    
    rgba_ice[ice_mask, 0] = np.clip(base_r * (1-alpha) + ice_r * alpha, 0, 255).astype(np.uint8)
    rgba_ice[ice_mask, 1] = np.clip(base_g * (1-alpha) + ice_g * alpha, 0, 255).astype(np.uint8)
    rgba_ice[ice_mask, 2] = np.clip(base_b * (1-alpha) + ice_b * alpha, 0, 255).astype(np.uint8)

Image.fromarray(rgba_ice).save(OUT / "texture_ice_glow.png", optimize=True)
print(f"  ✓ Ice overlay texture saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CPR Radar
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating CPR radar texture...")
import matplotlib.pyplot as plt
cpr_norm = np.clip(cpr_s / 3.0, 0, 1)
cmap = plt.cm.inferno
rgba_cpr = (cmap(cpr_norm) * 255).astype(np.uint8)
rgba_cpr[cpr_s <= 0.001] = [15, 12, 30, 255]
Image.fromarray(rgba_cpr).save(OUT / "texture_cpr.png", optimize=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Heightmap + Normal map
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating heightmap + normal map...")
Image.fromarray((dem_norm * 255).astype(np.uint8), mode='L').save(OUT / "heightmap_8bit.png")

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

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Data texture (for click-inspect)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating data texture...")
data_rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
data_rgba[:, :, 0] = np.clip(cpr_s / 3.0 * 255, 0, 255).astype(np.uint8)
data_rgba[:, :, 1] = np.clip(prob_s * 255, 0, 255).astype(np.uint8)
data_rgba[:, :, 2] = np.clip(slope_s / 45.0 * 255, 0, 255).astype(np.uint8)
data_rgba[:, :, 3] = (psr_s > 0.3).astype(np.uint8) * 255
Image.fromarray(data_rgba).save(OUT / "data_texture.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Metadata
# ═══════════════════════════════════════════════════════════════════════════════
meta = {
    "dem_min_m": dem_min, "dem_max_m": dem_max, "dem_range_m": dem_max - dem_min,
    "grid_size": SIZE,
    "pixel_size_m": round(25 * (H / SIZE), 1),
    "total_width_km": round(25 * W / 1000, 1),
    "total_height_km": round(25 * H / 1000, 1),
    "source": "LROC WAC South Pole Summer Mosaic (128 ppd) + LOLA DEM",
    "craters": [
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
}
with open(OUT / "terrain_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"Done in {time.time()-t0:.0f}s — Using REAL LROC WAC photography!")
print(f"{'='*60}")
for p in sorted(OUT.glob("*")):
    if p.suffix in ('.png', '.json'):
        print(f"  {p.name:30s} {p.stat().st_size/1e6:.2f} MB")
