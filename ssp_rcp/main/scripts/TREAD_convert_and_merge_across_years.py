import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

csv_files = snakemake.input

if not csv_files:
    raise ValueError(f"No CSV files found for {snakemake.wildcards.clim_factor}")

out_path = snakemake.output[0]
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

first = True

for f in csv_files:
    print("Reading:", f)
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError(f"Empty CSV: {f}")

    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in {f}")

    time_cols = [c for c in df.columns if c not in ["LON", "LAT"] and c.isdigit()]
    valid_time_cols = []
    for c in time_cols:
        try:
            pd.to_datetime(c, format="%Y%m%d")
            valid_time_cols.append(c)
        except ValueError:
            print(f"Skipping invalid date column: {c}")

    times = pd.to_datetime(valid_time_cols, format="%Y%m%d")

    lon = np.sort(df["LON"].unique())
    lat = np.sort(df["LAT"].unique())

    data = np.full(
        (len(times), len(lat), len(lon)),
        np.nan,
        dtype=np.float32,  # optional but strongly recommended
    )

    for i, tcol in enumerate(valid_time_cols):
        grid = df.pivot_table(values=tcol, index="LAT", columns="LON")
        grid = grid.reindex(index=lat, columns=lon)
        data[i] = grid.values

    parts = Path(f).name.split("_")
    year = parts[-1].split(".")[0]

    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name=snakemake.wildcards.clim_factor,
        attrs={"year": year},
    )

    ds = da.to_dataset()

    ds.to_netcdf(
    out_path,
    mode="a" if not first else "w",
    unlimited_dims=["time"],
    engine="netcdf4",
)


print("Finished writing:", out_path)
