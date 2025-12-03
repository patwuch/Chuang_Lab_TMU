from snakemake.io import glob_wildcards
import glob
from pathlib import Path

def find_files(wc):
    pattern = str(AR6_RAW_DIR / wc.var / wc.ssp / "*.csv")
    files = glob.glob(pattern)
    # keep only those matching model
    return [f for f in files if f"_{wc.model}_" in os.path.basename(f)]

raw_paths = glob_wildcards(str(AR6_RAW_DIR / "{var}/{ssp}/{fname}.csv"))
all_vars   = sorted(set(raw_paths.var))
all_ssps   = sorted(set(raw_paths.ssp))

# Extract models from filenames
all_models = sorted({
    f.split("_")[-2]    # model is second-to-last element
    for f in raw_paths.fname
})

# -------------------------------------------
# Step 1: Build one NetCDF per (var, ssp, model)
# -------------------------------------------
rule merge_ar6_csv_to_netcdf:
    input:
        lambda wc: [
            f for f in glob.glob(str(AR6_RAW_DIR / wc.var / wc.ssp / "*.csv"))
            if f"_{wc.model}_" in os.path.basename(f)
        ]
    output:
        AR6_NETCDF_DIR / "{var}" / "{ssp}" / "{model}.nc"
    run:
        import pandas as pd, numpy as np, xarray as xr, os

        all_da = []

        for f in input:
            df = pd.read_csv(f)
            if df.empty:
                continue

            # --- metadata extraction from filename ---
            fname = os.path.basename(f)
            parts = fname.split("_")

            ssp = next((p for p in parts if p.startswith("ssp")), "unknown")
            model = parts[parts.index(ssp) + 1] if ssp in parts else "unknown"
            year = parts[-1].split(".")[0]

            # --- build array ---
            time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]
            times = pd.to_datetime(time_cols, format="%Y%m%d", errors="coerce")

            lon = np.sort(df["LON"].unique())
            lat = np.sort(df["LAT"].unique())
            data = np.full((len(times), len(lat), len(lon)), np.nan)

            for i, tcol in enumerate(time_cols):
                grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
                grid = grid.reindex(index=lat, columns=lon)
                data[i] = grid.values

            da = xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": times, "lat": lat, "lon": lon},
                name=wildcards.var,
                attrs={"ssp": ssp, "model": model, "year": year},
            )

            all_da.append(da)

        ds = xr.concat(all_da, dim="time").sortby("time")
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        ds.to_netcdf(output[0])

# Create all per-model targets dynamically
rule all_ar6_models:
    input:
        expand(
            AR6_NETCDF_DIR / "{var}" / "{ssp}" / "{model}.nc",
            var = all_vars,
            ssp = all_ssps,
            model = all_models
        )

# -----------------------------
# Step 2: (Optional) Combine per-model NetCDF → per-variable NetCDF
# -----------------------------
rule combine_ar6_netcdf:
    input:
        lambda wc: sorted(glob.glob(str(AR6_NETCDF_DIR / wc.var / "*" / "*.nc")))
    output:
        AR6_NETCDF_DIR / "{var}.nc"
    run:
        import xarray as xr, pandas as pd, os

        if not input:
            print(f"No NetCDF files to combine for var={wildcards.var}")
            return

        print(f"Combining {len(input)} files for var={wildcards.var}")

        ds_list = []
        ssp_list = []
        model_list = []

        # Open each per-model NetCDF and extract SSP and model
        for f in input:
            ds = xr.open_dataset(f)
            ssp = ds.attrs.get("ssp", "unknown_ssp")
            model = ds.attrs.get("model", "unknown_model")

            # Add coordinates for ssp and model
            ds = ds.expand_dims({"ssp": [ssp], "model": [model]})
            ds_list.append(ds)
            ssp_list.append(ssp)
            model_list.append(model)

        # Concatenate along ssp and model
        combined = xr.concat(ds_list, dim=["ssp", "model"], combine_attrs="override")

        # Fill missing values with NaN
        combined = combined.fillna(float("nan"))

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        combined.to_netcdf(output[0])
        print(f"Saved combined variable NetCDF: {output[0]}")

rule all_ar6_variables:
    input:
        expand(AR6_NETCDF_DIR / "{var}.nc", var=all_vars)

# -------------------------------------------
# Step 3: Merge across all variables
# -------------------------------------------
rule make_ar6_all_netcdf:
    input:
        lambda wc: sorted(glob.glob(str(AR6_NETCDF_DIR / "*.nc")))
    output:
        NETCDF_DIR / "ar6_all.nc"
    run:
        import xarray as xr, os

        if not input:
            raise ValueError("No input NetCDF files found for merging!")

        print(f"Merging {len(input)} variable NetCDFs into final ar6_all.nc")

        ds_list = []
        var_names = []

        for f in input:
            ds = xr.open_dataset(f)
            var_name = ds[list(ds.data_vars)[0]].name  # get the main variable name
            ds = ds.expand_dims({"variable": [var_name]})
            ds_list.append(ds)
            var_names.append(var_name)

        # Concatenate along variable dimension
        combined = xr.concat(ds_list, dim="variable", combine_attrs="override")

        # Fill missing entries with NaN
        combined = combined.fillna(float("nan"))

        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        combined.to_netcdf(output[0])
        print(f"Saved final merged NetCDF: {output[0]}")
