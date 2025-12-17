import xarray as xr
import numpy as np

# Load dataset
ds = xr.open_dataset("/home/patwuch/Documents/projects/Chuang_Lab_TMU/ssp_rcp/work/data/processed/NetCDF/GWL_NetCDF/gwl_all.nc")

# Function to check masking
def check_fill_values(ds, fill_candidates=[-99.9, -9999]):
    for var in ds.data_vars:
        values = ds[var].values
        masked = np.isnan(values)
        n_masked = masked.sum()
        n_fill = np.isin(values, fill_candidates).sum()
        
        print(f"Variable: {var}")
        print(f"  Total elements: {values.size}")
        print(f"  Masked (NaN) count: {n_masked}")
        print(f"  Raw fill candidate count: {n_fill}")
        print("  ---")
        
check_fill_values(ds)
