import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from collections import defaultdict

# -------------------------
# Configuration
# -------------------------
AR6_RAW_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/raw/AR6")       # adjust paths
AR6_NETCDF_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/processed/NetCDF/AR6")
NETCDF_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/processed/NetCDF")

# -------------------------
# Step 0: Discover all files
# -------------------------
raw_paths = glob.glob(str(AR6_RAW_DIR / "*" / "*" / "*.csv"))

all_vars = sorted({Path(f).parts[-3] for f in raw_paths})
all_ssps = sorted({Path(f).parts[-2] for f in raw_paths})
all_models = sorted({
    Path(f).stem.split("_")[-2] for f in raw_paths
})

print(f"Found variables: {all_vars}")
print(f"Found SSPs: {all_ssps}")
print(f"Found models: {all_models}")

# =============================================================================
# CLEANING FUNCTION
# =============================================================================
def clean_ds(ds):
    """Remove duplicate time stamps and sort."""
    if "time" not in ds:
        return ds

    t = ds.time.to_index()

    if t.has_duplicates:
        print("  -> Removing duplicate timestamps")
        ds = ds.sel(time=~t.duplicated())

    ds = ds.sortby("time")
    return ds


# =============================================================================
# STEP 1 — CLEAN ALL PER-MODEL NETCDF FILES
# =============================================================================

print("\n================ CLEANING PER-MODEL NETCDF FILES ================\n")

for var in all_vars:
    for ssp in all_ssps:
        pattern = AR6_NETCDF_DIR / var / ssp / "*.nc"
        files = sorted(glob.glob(str(pattern)))
        if not files:
            continue

        # for f in files:
        #     print(f"CLEANING: {f}")

        #     with xr.open_dataset(f) as ds:
        #         ds_clean = clean_ds(ds)
        #         ds_clean.load()  # fully read into memory

        #     ds_clean.to_netcdf(f)
        #     print(f"  -> cleaned and saved")
import xarray as xr
from collections import defaultdict
import glob

print("\n===== COMBINING MODELS INTO PER-VARIABLE NETCDF =====\n")

for var in all_vars:
    print(f"\nProcessing variable: {var}")
    var_dir = AR6_NETCDF_DIR / var
    files = glob.glob(str(var_dir / "*" / "*.nc"))
    if not files:
        print(f"No files found for variable {var}")
        continue

    # Group datasets by SSP
    ssp_groups = defaultdict(list)

    for f in files:
        # Lazy open with dask chunks to avoid loading all into memory
        ds = xr.open_dataset(f, chunks={"time": 10})  # adjust chunk size if needed
        ds = clean_ds(ds)

        ssp = ds.attrs.get("ssp", "historical")
        model = ds.attrs.get("model", "unknown_model")

        ds_expanded = ds.expand_dims({"ssp": [ssp], "model": [model]})
        ssp_groups[ssp].append(ds_expanded)

    # Concat models within each SSP lazily
    ssp_datasets = []
    for ssp, group in ssp_groups.items():
        if len(group) == 0:
            continue
        print(f"  + Combining {len(group)} models for SSP {ssp}")
        ds_ssp = xr.concat(group, dim="model", combine_attrs="override", coords="minimal")
        ssp_datasets.append(ds_ssp)

    if len(ssp_datasets) == 0:
        print(f"No valid SSP datasets for variable {var}")
        continue

    # Concat across SSPs
    combined = xr.concat(ssp_datasets, dim="ssp", combine_attrs="override", coords="minimal")
    combined = combined.sortby("time")

    # Save final combined file (this triggers computation)
    out_path = AR6_NETCDF_DIR / f"{var}.nc"
    combined.to_netcdf(out_path)
    print(f"✔ Saved combined variable file → {out_path}")

# -------------------------
# Step 3: Merge all variables → final NetCDF
# -------------------------
files = sorted(glob.glob(str(AR6_NETCDF_DIR / "*.nc")))
if not files:
    raise ValueError("No variable NetCDF files found for merging!")

ds_list = []
for f in files:
    ds = xr.open_dataset(f)
    var_name = list(ds.data_vars)[0]
    ds = ds.expand_dims({"variable": [var_name]})
    ds_list.append(ds)

combined_all = xr.concat(ds_list, dim="variable", combine_attrs="override")
combined_all = combined_all.fillna(float("nan"))

out_path = NETCDF_DIR / "ar6_all.nc"
os.makedirs(out_path.parent, exist_ok=True)
combined_all.to_netcdf(out_path)
print(f"Saved final merged NetCDF: {out_path}")
