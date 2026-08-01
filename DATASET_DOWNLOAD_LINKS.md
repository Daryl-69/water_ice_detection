# 📦 Dataset Download Links — Lunar South Pole Ice Detection

Save this file. If you ever delete the datasets to free space, you can re-download everything from these links.

**Total size: ~53.7 GB**

---

## 1. ISRO Chandrayaan-2 DFSAR (Dual-Frequency SAR)

> Source: ISRO PRADAN (https://pradan.issdc.gov.in)
> You need to log in with your ISRO PRADAN account to download these.

### Cross-Polarization (20 GB)
- **File**: `ch2_sar_ncxl_20251022t044533259_d_cp_d18_Bundle.tar`
- **Portal**: https://pradan.issdc.gov.in/ch2/
- **How to find**: Search for DFSAR → Cross-Pol (CP) → South Pole region → Date: 2025-10-22
- **Product ID**: `ch2_sar_ncxl_20251022t044533259_d_cp_d18`

### Full-Polarization (22.4 GB)  
- **File**: `ch2_sar_ncxl_20210228t212710499_d_fp_d32_Bundle.tar`
- **Portal**: https://pradan.issdc.gov.in/ch2/
- **How to find**: Search for DFSAR → Full-Pol (FP) → South Pole region → Date: 2021-02-28
- **Product ID**: `ch2_sar_ncxl_20210228t212710499_d_fp_d32`

### DFSAR Mosaic (2.6 GB)
- **File 1**: `ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx.zip` (0.54 GB)
- **File 2**: `ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx_Bundle.tar` (1.29 GB)
- **File 3**: `ch2_sar_ndxl_20250630my4rspeast_d_fp_xxx.zip` (0.74 GB)
- **Portal**: https://pradan.issdc.gov.in/ch2/
- **How to find**: Search for DFSAR Mosaic → South Pole East → Date: 2025-06-30

---

## 2. NASA LRO LOLA (Lunar Orbiter Laser Altimeter)

> Source: NASA PDS Geosciences Node

### LOLA South Pole DEM & Derived Products (~1 GB)

| File | Size | Direct Download |
|------|------|----------------|
| `Haworth_DEM_5mpp.tif` | 37 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Haworth_final_adj_5mpp_slp.tif` | 136 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Haworth_final_adj_5mpp_slperr.tif` | 136 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Haworth_final_adj_5mpp_surf.tif` | 136 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Haworth_final_adj_5mpp_toterr.tif` | 136 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Shoemaker_DEM_5mpp.tif` | 31 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Shoemaker_final_adj_5mpp_slp.tif` | 61 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Shoemaker_final_adj_5mpp_slperr.tif` | 61 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Shoemaker_final_adj_5mpp_surf.tif` | 61 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `Shoemaker_final_adj_5mpp_toterr.tif` | 61 MB | https://pgda.gsfc.nasa.gov/products/90 |
| `LM7_final_adj_5mpp_surf.tif` | 61 MB | https://pgda.gsfc.nasa.gov/products/90 |

> **Alternative**: All LOLA South Pole DEMs are available at:
> https://pgda.gsfc.nasa.gov/products/90
> Search for "South Pole" DEMs at 5m/pixel resolution.

### LOLA PSR (Permanently Shadowed Regions) Map
| File | Size | Direct Download |
|------|------|----------------|
| `LPSR_85S_060M_201608.JP2` | 5 MB | https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_shadr/ |
| `LPSR_85S_060M_201608.LBL` | <1 KB | (same directory as above) |

### WAC South Pole Mosaic
| File | Size | Direct Download |
|------|------|----------------|
| `WAC_SOUTH_POLE.TIF` | 141 MB | https://wms.lroc.asu.edu/lroc/view_rdr_product/WAC_POLE_MOSAIC_S |

> **Note**: The WAC mosaic can also be found at:
> https://astrogeology.usgs.gov/search?pmi-target=moon
> Or directly from LROC: https://wms.lroc.asu.edu/lroc/

---

## 3. NASA LRO Mini-RF (Miniature Radio Frequency)

### Global CPR Map (4 GB)
| File | Size | Direct Download |
|------|------|----------------|
| `global_cpr_128ppd_simp_0c.img` | 3.96 GB | https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-shadr-v1/lromrf_1xxx/data/ |

> **Portal**: PDS Geosciences Node → LRO → Mini-RF
> https://pds-geosciences.wustl.edu/missions/lro/mrf.htm
> Look for the "Global CPR Map" at 128 pixels/degree, simple cylindrical projection.

---

## 4. Reference Paper

| File | Size | Source |
|------|------|--------|
| `Bhiravarasu_2021_Planet._Sci._J._2_134.pdf` | 5 MB | https://doi.org/10.3847/PSJ/ac12d1 |

---

## Quick Re-Download Script (PowerShell)

```powershell
# Create dataset directories
$base = "D:\1_p3-isro\datasets"
mkdir "$base\lola" -Force
mkdir "$base\minirf" -Force
# DFSAR requires manual download from PRADAN portal (login required)

# Mini-RF Global CPR (check exact URL on PDS site)
# Invoke-WebRequest "https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-shadr-v1/lromrf_1xxx/data/global_cpr_128ppd_simp_0c.img" -OutFile "$base\minirf\global_cpr_128ppd_simp_0c.img"

# LOLA & WAC — download manually from the portal links above
# DFSAR — must download from https://pradan.issdc.gov.in/ch2/ with login
```

> [!IMPORTANT]
> **DFSAR data requires ISRO PRADAN login** — there's no direct URL. Bookmark https://pradan.issdc.gov.in/ch2/ and use the Product IDs listed above to find the exact files.

> [!TIP]
> The LOLA and Mini-RF data are publicly available with no login. The DFSAR data is the hardest to re-download since it requires the ISRO portal.
