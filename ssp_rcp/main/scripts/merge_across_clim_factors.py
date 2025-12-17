import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# Fill value to be masked
FILL_VALUE = -99.9

# Load datasets
ds_list = [xr.open_dataset(f) for f in snakemake.input]

# Mask fill values in each dataset before merging
for i, ds in enumerate(ds_list):
    for var in ds.data_vars:
        ds[var] = ds[var].where(ds[var] != FILL_VALUE)
    ds_list[i] = ds  # update list with masked dataset

# Merge datasets
merged = xr.merge(ds_list)

# Make sure output directory exists
Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)

# Save merged dataset
merged.to_netcdf(snakemake.output[0])
print("All processed!")
