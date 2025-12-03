import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


GWL_RAW_DIR = Path("/home/patwuch/projects/chuang_lab_tmu/work/ssp_rcp/data/raw/GWL")
GWL_NETCDF_DIR = Path("netcdf/gwl")
NETCDF_DIR = Path("netcdf")

###############################################################################
# Helper discovery functions
###############################################################################

def find_clim_factors():
    """Return all climate factor directories inside the raw GWL folder."""
    return [p.name for p in GWL_RAW_DIR.iterdir() if p.is_dir()]

def find_gwls(clim_factor):
    """Return all GWL-level directories inside a given climate factor folder."""
    folder = GWL_RAW_DIR / clim_factor
    return [p.name for p in folder.iterdir() if p.is_dir()]
print(find_clim_factors())
for cf in find_clim_factors():
    print(cf, find_gwls(cf))

###############################################################################
# RULE: Merge CSVs → One NetCDF per (clim_factor, gwl)
###############################################################################

rule merge_gwl_csv_to_netcdf:
    # Wildcards come from output pattern:
    #   netcdf/gwl/<clim_factor>/<gwl>.nc
    input:
        lambda w: list((GWL_RAW_DIR / w.clim_factor / w.gwl).glob("*.csv"))
    output:
        GWL_NETCDF_DIR / "{clim_factor}" / "{gwl}.nc"
    run:
        print(f"Processing CSVs for {wildcards.clim_factor}/{wildcards.gwl}")

        all_da = []

        for f in input:
            df = pd.read_csv(f)
            if df.empty:
                raise ValueError(f"Empty CSV: {f}")

            dup = df.columns[df.columns.duplicated()]
            if len(dup) > 0:
                raise ValueError(f"Duplicate columns in {f}: {list(dup)}")

            # time parsing
            time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]
            times = pd.to_datetime(time_cols, format="%Y%m%d")

            lon = np.sort(df["LON"].unique())
            lat = np.sort(df["LAT"].unique())

            data = np.full((len(times), len(lat), len(lon)), np.nan)

            for i, tcol in enumerate(time_cols):
                grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
                grid = grid.reindex(index=lat, columns=lon)
                data[i] = grid.values

            # metadata extraction
            fname = f.name
            parts = fname.split("_")
            ssp = next((p for p in parts if p.startswith("ssp")), "unknown")
            model = parts[parts.index(ssp) + 1] if ssp in parts else "unknown"
            year = parts[-1].split(".")[0]

            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": times, "lat": lat, "lon": lon},
                name=wildcards.clim_factor,
                attrs={"ssp": ssp, "model": model, "gwl": wildcards.gwl, "year": year},
            )
            all_da.append(da)

        ds = xr.concat(all_da, dim="time").sortby("time").groupby("time").first()

        output[0].parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output[0])

###############################################################################
# RULE: Combine all GWL NetCDFs for each climate factor
###############################################################################

rule combine_gwl_netcdf:
    input:
        lambda w: sorted((GWL_NETCDF_DIR / w.clim_factor).glob("*.nc"))
    output:
        GWL_NETCDF_DIR / "combined_{clim_factor}.nc"
    run:
        ds_list = []

        for f in input:
            ds = xr.open_dataset(f)
            gwl_val = f.stem  # e.g. "GWL1.5"
            for var in ds:
                ds[var] = ds[var].expand_dims({"gwl": [gwl_val]})
            ds_list.append(ds)

        combined = xr.concat(ds_list, dim="gwl")
        output[0].parent.mkdir(parents=True, exist_ok=True)
        combined.to_netcdf(output[0])

###############################################################################
# RULE: Merge all climate factors into a single gwl_all.nc
###############################################################################

rule make_gwl_all_netcdf:
    input:
        lambda w: [GWL_NETCDF_DIR / f"combined_{cf}.nc" for cf in find_clim_factors()]
    output:
        NETCDF_DIR / "gwl_all.nc"
    run:
        ds_list = [xr.open_dataset(f) for f in input]
        merged = xr.merge(ds_list)
        output[0].parent.mkdir(parents=True, exist_ok=True)
        merged.to_netcdf(output[0])
        print("All processed!")

