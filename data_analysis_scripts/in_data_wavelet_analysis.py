# %%
# Import basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os 
from pathlib import Path
import sys
from tqdm import tqdm
import pycwt as wavelet
from pycwt import wct_significance
from pycwt import helpers
import matplotlib.dates as mdates
import scipy
from scipy.signal import detrend
from matplotlib.ticker import LogLocator, FormatStrFormatter

# Define project root based on notebook location
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve()  # fallback

PROJECT_ROOT = find_project_root(Path.cwd())
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "IN" / '20250626'


# %% [markdown]
# Load in environmental and dengue fever files.

# %%
monthly_env_national = pd.read_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_env_national_2010_2023.csv')
monthly_env_region = pd.read_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_env_region_2010_2023.csv')
monthly_env_id = pd.read_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_env_id_2010_2023.csv')
dengue_by_ID_2 = pd.read_csv(PROCESSED_DIR / 'INDONESIA' / 'dengue_processed_data.csv')

# Melt the DataFrame to transform monthly infection columns into rows
df_melted = dengue_by_ID_2.melt(
    id_vars=['ID_2', 'Region', 'Year', 'Population'],
    value_vars=[f'Infection_{i}' for i in range(1, 13)],
    var_name='Month_num',
    value_name='Monthly_Infection'
)
print(df_melted.columns)
# Extract month number and convert to int
df_melted['Month_num'] = df_melted['Month_num'].str.extract(r'(\d+)').astype(int)

# Create 'YearMonth' column
df_melted['YearMonth'] = pd.to_datetime(
    df_melted['Year'].astype(str) + '-' + df_melted['Month_num'].astype(str).str.zfill(2)
)

# Group by ID_2 and YearMonth
df_agg = df_melted.groupby(['ID_2', 'Region', 'YearMonth']).agg(
    Monthly_Infection=('Monthly_Infection', 'sum'),
    Population=('Population', 'first')  # assuming population per ID_2 doesn't change monthly
).reset_index()
print(df_agg.columns)
# Calculate incidence rate
df_agg['Incidence_Rate'] = (df_agg['Monthly_Infection'] / df_agg['Population']) * 100000

# Ensure environmental data has datetime type for YearMonth
monthly_env_id['YearMonth'] = pd.to_datetime(monthly_env_id['YearMonth'])
print(monthly_env_id.columns)

# Merge and keep Region only from the dengue data
# Use suffixes to differentiate the 'Region' columns during merge
monthly_dengue_env_id = pd.merge(
    df_agg, 
    monthly_env_id, 
    on=['ID_2', 'YearMonth'], 
    how='inner',
    suffixes=('_dengue', '_env') # Add suffixes to overlapping column names
)

# Keep the 'Region' column from the dengue data (df_agg) and drop the one from env data
# Rename 'Region_dengue' to 'Region' and drop 'Region_env'
monthly_dengue_env_id.rename(columns={'Region_dengue': 'Region'}, inplace=True)
monthly_dengue_env_id.drop(columns=['Region_env'], inplace=True, errors='ignore')

print(monthly_dengue_env_id.columns)
# Fill missing Monthly_Infection or Population with 0 (if any)
monthly_dengue_env_id['Monthly_Infection'] = monthly_dengue_env_id['Monthly_Infection'].fillna(0)
monthly_dengue_env_id['Population'] = monthly_dengue_env_id['Population'].fillna(0)

# Recalculate Incidence Rate
monthly_dengue_env_id['Incidence_Rate'] = (
    (monthly_dengue_env_id['Monthly_Infection'] / monthly_dengue_env_id['Population'].replace(0, pd.NA)) * 100000
).fillna(0)

# Save final merged data
monthly_dengue_env_id.to_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_dengue_env_id.csv', index=False)

print("\nFinal columns after saving:")
print(pd.read_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_dengue_env_id.csv').columns)

# %%
# Melt the DataFrame to transform monthly infection columns into rows
# df_melted = dengue_by_ID_2.melt(
#     id_vars=['ID_2', 'Region', 'Year', 'Population'],
#     value_vars=[f'Infection_{i}' for i in range(1, 13)],
#     var_name='Month_num',
#     value_name='Monthly_Infection'
# )
# # Extract month number and convert to string for 'YearMonth'
# df_melted['Month_num'] = df_melted['Month_num'].str.extract(r'(\d+)').astype(int)
# # Create 'YearMonth' column
# df_melted['YearMonth'] = df_melted['Year'].astype(str) + '-' + df_melted['Month_num'].astype(str).str.zfill(2)
# # Aggregate data by 'Region' and 'YearMonth'
# df_agg = df_melted.groupby(['Region', 'YearMonth']).agg(
#     Monthly_Infection=('Monthly_Infection', 'sum'),
#     Population=('Population', 'sum')
# ).reset_index()
# # Calculate 'Incidence_Rate'
# df_agg['Incidence_Rate'] = (df_agg['Monthly_Infection'] / df_agg['Population']) * 100000
# df_agg['YearMonth'] = pd.to_datetime(df_agg['YearMonth'])
# monthly_env_region['YearMonth'] = pd.to_datetime(monthly_env_region['YearMonth'])
# # Merge df_agg with env_var
# monthly_dengue_env_region = pd.merge(df_agg, monthly_env_region, on=['Region', 'YearMonth'], how='inner')
# monthly_dengue_env_region.to_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_dengue_env_region.csv', index=False)

# # %%
# # Melt the DataFrame
# df_melted = dengue_by_ID_2.melt(
#     id_vars=['ID_2', 'Region', 'Year', 'Population'],
#     value_vars=[f'Infection_{i}' for i in range(1, 13)],
#     var_name='Month_num',
#     value_name='Monthly_Infection'
# )
# # Extract numeric month
# df_melted['Month_num'] = df_melted['Month_num'].str.extract(r'(\d+)').astype(int)

# # Create YearMonth
# df_melted['YearMonth'] = df_melted['Year'].astype(str) + '-' + df_melted['Month_num'].astype(str).str.zfill(2)
# df_melted['YearMonth'] = pd.to_datetime(df_melted['YearMonth'])

# # Aggregate at the national level
# df_agg_national = df_melted.groupby(['Year', 'YearMonth']).agg(
#     Monthly_Infection=('Monthly_Infection', 'sum'),
#     Population=('Population', 'sum')  # Total national population for that year
# ).reset_index()

# # Calculate Incidence Rate
# df_agg_national['Incidence_Rate'] = (df_agg_national['Monthly_Infection'] / df_agg_national['Population']) * 100000

# # Convert YearMonth in env dataframe
# monthly_env_national['YearMonth'] = pd.to_datetime(monthly_env_national['YearMonth'])

# # Merge with national-level environmental data
# monthly_dengue_env_national = pd.merge(df_agg_national, monthly_env_national, on='YearMonth', how='inner')

# # Save result
# monthly_dengue_env_national.to_csv(PROCESSED_DIR / 'INDONESIA' / 'monthly_dengue_env_national.csv', index=False)


# # %% [markdown]
# # Below is code for conducting wavelet coherence analysis between national incidence rate and national environmental variable.

# # %%
# # Convert integers to specific version of numpy that works with scipy detrend
# if not hasattr(np, 'int'):
#     np.int = int

# # Create copy of data and set 
# df = monthly_dengue_env_national.copy()

# def plot_wavelet_coherence(
#     unit_df, 
#     incidence_detrend,
#     aWCT,
#     WCT,
#     coi,
#     freqs,
#     sig,
#     incidence_col,
#     target_env,
#     unit_name, 
#     SAVE_DIR=FIG_DIR,
#     time_col='Date'
# ):
#     # Construct uniform time index
#     unit_start_date = pd.to_datetime(unit_df[time_col].min()) 
#     uniform_time_index = pd.date_range(start=unit_start_date, periods=len(incidence_detrend), freq='MS')
    
#     n_time_points_wct = WCT.shape[1]
#     plot_time = uniform_time_index[:n_time_points_wct]
#     plot_coi = coi[:n_time_points_wct]
#     periods = 1 / freqs


#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Plot base wavelet coherence
#     levels = np.linspace(0, 1, 100)
#     im = ax.contourf(plot_time, periods, WCT, levels=levels, cmap='jet', extend='both')
#     fig.colorbar(im, ax=ax, label='Wavelet Coherence')

#     # Debugging statements for significance contour
#     print(f"--- Debugging sig contour for {unit_name} - {incidence_col} vs {target_env} ---") # Renamed
#     print(f"   Shape of WCT: {WCT.shape}")
#     print(f"   Shape of sig (from wavelet.wct): {sig.shape}")
#     # Create sig_matrix as it would be used in the plot
#     sig_matrix = np.tile(sig[:, np.newaxis], (1, n_time_points_wct))

#     # Now use sig_matrix for plot
#     print(f"   Shape of sig_matrix (after tiling): {sig_matrix.shape}")
#     print(f"   Min/Max of WCT: {np.min(WCT):.4f} / {np.max(WCT):.4f}")
#     print(f"   Min/Max of sig (significance levels): {np.min(sig):.4f} / {np.max(sig):.4f}")
#     print(f"   Max value in sig_matrix (the threshold): {np.max(sig_matrix):.4f}")
#     # Check if any coherence value is above its corresponding significance threshold
#     print(f"   Are there any values in WCT >= sig_matrix? {(WCT >= sig_matrix).any()}")
#     print(f"------------------------------------------------------------------")


#     # Overlay 95% significance contour
#     ax.contour(
#         plot_time, periods, WCT,
#         levels=[sig_matrix.max()],
#         colors='k',
#         linewidths=0.5,
#         linestyles='dashed',
#         zorder=5
#     )

#     # Highlight significant coherence regions with hatching
#     sig_mask = WCT >= sig_matrix
#     ax.contourf(
#         plot_time,
#         periods,
#         sig_mask,
#         levels=[0.5, 1],
#         colors='none',
#         hatches=['', '////'],
#         zorder=4
#     )

#     # Shade cone of influence
#     ax.fill_between(plot_time, plot_coi, periods.max(), color='gray', alpha=0.5,
#                     hatch='x', zorder=2)

#     # Plot phase arrows
#     k = 5
#     X, Y = np.meshgrid(plot_time[::k], periods[::k])
#     angle = 0.5 * np.pi - aWCT[::k, ::k]
#     U_raw, V_raw = np.cos(angle), np.sin(angle)

#     # Mask arrows in the cone of influence
#     coi_mask = (periods[::k, np.newaxis] > plot_coi[::k]).T
#     U = np.ma.array(U_raw, mask=coi_mask)
#     V = np.ma.array(V_raw, mask=coi_mask)

#     # Robust magnitude calculation (optional but safe for coloring or inspection)
#     with np.errstate(invalid='ignore'):
#         norm = np.sqrt(U**2 + V**2)
#         norm = np.ma.masked_where(norm == 0, norm)  # Optional safeguard

#     # Plot arrows with original magnitude (not normalized)
#     ax.quiver(
#         X, Y, U, V,
#         pivot='middle',
#         scale_units='width',
#         scale=None,  # Let matplotlib scale arrows based on magnitude
#         width=0.004,
#         headwidth=2,
#         headlength=3,
#         headaxislength=2,
#         minlength=0.05,
#         color='k',
#         zorder=3
#     )


#     # Axis formatting
#     ax.set_xlabel("Time (Year)")
#     ax.set_ylabel("Period (Months)")
#     ax.set_title(f"Wavelet Coherence: {unit_name} - {incidence_col} vs {target_env}") # Renamed
#     ax.set_yscale('log', base=2)
#     ax.set_ylim([periods.min(), periods.max()])
#     ax.invert_yaxis()

#     possible_ticks = [2, 4, 8, 16, 32, 64, 128]
#     yticks = [p for p in possible_ticks if periods.min() <= p <= periods.max()]
#     ax.set_yticks(yticks)
#     ax.set_yticklabels([str(p) for p in yticks])

#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#     ax.xaxis.set_major_locator(mdates.YearLocator())

#     plt.tight_layout()

#     # Save the figure
#     filename = f"WCT_{unit_name.replace(' ', '_')}_{incidence_col}_vs_{target_env}.png" # Renamed & sanitized for filename
#     filepath = os.path.join(SAVE_DIR, filename)
#     plt.savefig(filepath, dpi=300)
#     plt.close(fig)
#     print(f"Plot saved: {filepath}")

# # Main execution for a single geographical unit, comment out to select specific ones to plot
# env_list = [
#     'temperature_2m', 'temperature_2m_min',
#     'temperature_2m_max', 'precipitation', 'potential_evaporation_sum',
#     'total_evaporation_sum', 'ANOM1+2',  'ANOM3',
#      'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East', 'evaporative_stress_index', 'aridity_index'
# ]

# # Designate region, infection, time column. 
# # Comment out either infection_col or incidence_col based on which statistic to plot
# # infection_col = 'Monthly_Infection_SUM' 
# incidence_col = 'Incidence_Rate'  # Use incidence rate for wavelet coherence
# time_col = 'YearMonth'


# single_unit_df = df.copy() 
# single_unit_name = "Indonesia" # Or any appropriate name for your single unit

# mother = wavelet.Morlet()

# full_incidence = single_unit_df[incidence_col].values

# # Use tqdm for the environmental variables loop
# from tqdm.auto import tqdm

# for target_env in tqdm(env_list, desc=f"Processing Envs for {single_unit_name}"):
#     if target_env not in single_unit_df.columns:
#         print(f"Error: {target_env} not found for {single_unit_name}. Skipping.")
#         continue

#     full_env = single_unit_df[target_env].values

#     print("Detrending signals...")
#     incidence_detrend = detrend(full_incidence).flatten()
#     environment_detrend = detrend(full_env).flatten()

#     dt = 1      # monthly time step
#     dj = 1 / 12
#     s0 = 2 * dt

#     try:
#         print(f"Computing wavelet coherence for {single_unit_name} - {target_env}...")
#         WCT, aWCT, coi, freqs, sig = wavelet.wct(
#             incidence_detrend,
#             environment_detrend,
#             dt=dt,
#             dj=dj,
#             s0=s0,
#             wavelet=mother,
#             normalize=False,
#             mc_count=1000, 
#             cache=False
#         )
#         print("Wavelet coherence computed.")
#         print(f"\n--- Properties of Wavelet Coherence Variables for {single_unit_name} - {target_env} ---")
#         print(f"aWCT (Phase Angle) properties:")
#         print(f"   Shape: {aWCT.shape}")
#         print(f"   Data Type: {aWCT.dtype}")
#         print(f"   Min Value: {np.min(aWCT):.4f}")
#         print(f"   Max Value: {np.max(aWCT):.4f}")
#         print(f"   Mean Value: {np.mean(aWCT):.4f}")
#         print(f"   Contains NaN: {np.isnan(aWCT).any()}")
#         print(f"   Contains Inf: {np.isinf(aWCT).any()}")

#         print(f"\nWCT (Coherence) properties:")
#         print(f"   Shape: {WCT.shape}")
#         print(f"   Data Type: {WCT.dtype}")
#         print(f"   Min Value: {np.min(WCT):.4f}")
#         print(f"   Max Value: {np.max(WCT):.4f}")
#         print(f"   Mean Value: {np.mean(WCT):.4f}")
#         print(f"   Contains NaN: {np.isnan(WCT).any()}")
#         print(f"   Contains Inf: {np.isinf(WCT).any()}")

#         print(f"\ncoi (Cone of Influence) properties:")
#         print(f"   Shape: {coi.shape}")
#         print(f"   Data Type: {coi.dtype}")
#         print(f"   Min Value: {np.min(coi):.4f}")
#         print(f"   Max Value: {np.max(coi):.4f}")
#         print(f"   Mean Value: {np.mean(coi):.4f}")
#         print(f"   Contains NaN: {np.isnan(coi).any()}")
#         print(f"   Contains Inf: {np.isinf(coi).any()}")

#         print(f"\nfreqs (Frequencies) properties:")
#         print(f"   Shape: {freqs.shape}")
#         print(f"   Data Type: {freqs.dtype}")
#         print(f"   Min Value: {np.min(freqs):.4f}")
#         print(f"   Max Value: {np.max(freqs):.4f}")
#         print(f"   Mean Value: {np.mean(freqs):.4f}")
#         print(f"   Contains NaN: {np.isnan(freqs).any()}")
#         print(f"   Contains Inf: {np.isinf(freqs).any()}")
#         print(f"   Corresponding Periods (Min/Max): {1/np.max(freqs):.2f} / {1/np.min(freqs):.2f} (Months)")

#         np.set_printoptions(threshold=np.inf, suppress=True)
#         print(f"\nsig (Significance Levels) full array:\n{sig}\n")
#         print(f"\nsig (Significance Levels) properties:")
#         print(f"   Shape: {sig.shape}")
#         print(f"   Data Type: {sig.dtype}")
#         print(f"   Min Value: {np.min(sig):.4f}")
#         print(f"   Max Value: {np.max(sig):.4f}")
#         print(f"   Mean Value: {np.mean(sig):.4f}")
#         print(f"   Contains NaN: {np.isnan(sig).any()}")
#         print(f"   Contains Inf: {np.isinf(sig).any()}")
#         if np.isnan(sig).any():
#             print("Warning: Replacing NaNs in `sig` with last valid value.")
#             nan_mask = np.isnan(sig)
#             last_valid = sig[~nan_mask][-1]
#             sig[nan_mask] = last_valid
#         print(f"------------------------------------------------------------------\n")
#         plot_wavelet_coherence(
#             unit_df=single_unit_df, # Pass the single unit DataFrame
#             incidence_detrend=incidence_detrend,
#             aWCT=aWCT,
#             WCT=WCT,
#             coi=coi,
#             freqs=freqs,
#             sig=sig,
#             incidence_col=incidence_col,
#             target_env=target_env,
#             unit_name=single_unit_name, # Pass the name of the single unit
#             SAVE_DIR=FIG_DIR,
#             time_col=time_col
#         )
#     except Exception as e:
#         print(f"Error during wavelet coherence or plotting for {single_unit_name}, {target_env}: {e}")

# # %% [markdown]
# # Below is code for conducting wavelet coherence analysis between incidence rate and environmental variable, region by region.

# # %%
# # Designate region, infection, time column
# incidence_col = 'Incidence_Rate'
# region_col = 'Region'
# time_col = 'YearMonth'
# # Create copy of data and set 
# df = monthly_dengue_env_region.copy()
# df = df.sort_values(by=[region_col, time_col]).reset_index(drop=True)

# def plot_wavelet_coherence(
#     region_df,
#     incidence_detrend,
#     aWCT,
#     WCT,
#     coi,
#     freqs,
#     sig,
#     incidence_col,
#     target_env,
#     target_region,
#     SAVE_DIR=FIG_DIR,
#     time_col='date'
# ):
#     # Construct uniform time index
#     region_start_date = pd.to_datetime(region_df[time_col].min())
#     uniform_time_index = pd.date_range(start=region_start_date, periods=len(incidence_detrend), freq='MS')
    
#     n_time_points_wct = WCT.shape[1]
#     plot_time = uniform_time_index[:n_time_points_wct]
#     plot_coi = coi[:n_time_points_wct]
#     periods = 1 / freqs


#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Plot base wavelet coherence
#     levels = np.linspace(0, 1, 100)
#     im = ax.contourf(plot_time, periods, WCT, levels=levels, cmap='jet', extend='both')
#     fig.colorbar(im, ax=ax, label='Wavelet Coherence')

#     # Debugging statements for significance contour (already present)
#     print(f"--- Debugging sig contour for {target_region} - {incidence_col} vs {target_env} ---")
#     print(f"   Shape of WCT: {WCT.shape}")
#     print(f"   Shape of sig (from wavelet.wct): {sig.shape}")
#     # Create sig_matrix as it would be used in the plot
#     sig_matrix = np.tile(sig[:, np.newaxis], (1, n_time_points_wct))

#     # Now use sig_matrix for plot
#     print(f"   Shape of sig_matrix (after tiling): {sig_matrix.shape}")
#     print(f"   Min/Max of WCT: {np.min(WCT):.4f} / {np.max(WCT):.4f}")
#     print(f"   Min/Max of sig (significance levels): {np.min(sig):.4f} / {np.max(sig):.4f}")
#     print(f"   Max value in sig_matrix (the threshold): {np.max(sig_matrix):.4f}")
#     # Check if any coherence value is above its corresponding significance threshold
#     print(f"   Are there any values in WCT >= sig_matrix? {(WCT >= sig_matrix).any()}")
#     print(f"------------------------------------------------------------------")


#     # Overlay 95% significance contour
#     # Assuming 'sig' directly provides the 95% significance level at each scale
#     ax.contour(
#     plot_time, periods, WCT,
#     levels=[sig_matrix.max()],  # or np.nanmean(sig), if safer
#     colors='k',
#     linewidths=0.5,
#     linestyles='dashed',
#     zorder=5
# )

#     # Highlight significant coherence regions with hatching
#     sig_mask = WCT >= sig_matrix # Mask where WCT is greater than or equal to the significance level
#     ax.contourf(
#         plot_time,
#         periods,
#         sig_mask,
#         levels=[0.5, 1], # Levels for the boolean mask: 0.5 to catch True values
#         colors='none',
#         hatches=['', '////'], # First hatch is empty for False, second for True
#         zorder=4
#     )

#     # Shade cone of influence
#     ax.fill_between(plot_time, plot_coi, periods.max(), color='gray', alpha=0.5,
#                     hatch='x', zorder=2)

#     # Plot phase arrows
#     k = 5
#     X, Y = np.meshgrid(plot_time[::k], periods[::k])
#     angle = 0.5 * np.pi - aWCT[::k, ::k]
#     U_raw, V_raw = np.cos(angle), np.sin(angle)

#     # Mask arrows in the cone of influence
#     coi_mask = (periods[::k, np.newaxis] > plot_coi[::k]).T
#     U = np.ma.array(U_raw, mask=coi_mask)
#     V = np.ma.array(V_raw, mask=coi_mask)

#     # Robust magnitude calculation (optional but safe for coloring or inspection)
#     with np.errstate(invalid='ignore'):
#         norm = np.sqrt(U**2 + V**2)
#         norm = np.ma.masked_where(norm == 0, norm)  # Optional safeguard

#     # Plot arrows with original magnitude (not normalized)
#     ax.quiver(
#         X, Y, U, V,
#         pivot='middle',
#         scale_units='width',
#         scale=None,  # Let matplotlib scale arrows based on magnitude
#         width=0.004,
#         headwidth=2,
#         headlength=3,
#         headaxislength=2,
#         minlength=0.05,
#         color='k',
#         zorder=3
#     )

#     # Axis formatting
#     ax.set_xlabel("Time (Year)")
#     ax.set_ylabel("Period (Months)")
#     ax.set_title(f"Wavelet Coherence: {target_region} - {incidence_col} vs {target_env}")
#     ax.set_yscale('log', base=2)
#     ax.set_ylim([periods.min(), periods.max()])
#     ax.invert_yaxis()

#     possible_ticks = [2, 4, 8, 16, 32, 64, 128]
#     yticks = [p for p in possible_ticks if periods.min() <= p <= periods.max()]
#     ax.set_yticks(yticks)
#     ax.set_yticklabels([str(p) for p in yticks])

#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#     ax.xaxis.set_major_locator(mdates.YearLocator())

#     plt.tight_layout()

#     # Save the figure
#     filename = f"WCT_{target_region}_{incidence_col}_vs_{target_env}.png"
#     filepath = os.path.join(SAVE_DIR, filename)
#     plt.savefig(filepath, dpi=300)
#     plt.close(fig)
#     print(f"Plot saved: {filepath}")

# # Main loop 
# env_list = [
#     'temperature_2m', 'temperature_2m_min',
#     'temperature_2m_max', 'precipitation', 'potential_evaporation_sum',
#     'total_evaporation_sum', 'ANOM1+2', 'ANOM3',
#     'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East', 'evaporative_stress_index', 'aridity_index'
# ]

# region_list = df[region_col].unique()
# mother = wavelet.Morlet()

# for target_region in tqdm(region_list, desc="Processing Regions"):
#     region_df = df[df[region_col] == target_region].copy()
#     if region_df.empty:
#         print(f"Error: Target region '{target_region}' not found in DataFrame. Skipping.")
#         continue

#     full_incidence = region_df[incidence_col].values

#     for target_env in tqdm(env_list, desc=f"Processing Envs for {target_region}", leave=False):
#         if target_env not in region_df.columns:
#             print(f"Error: {target_env} not found for {target_region}. Skipping.")
#             continue

#         full_env = region_df[target_env].values

#         print("Detrending signals...")
#         incidence_detrend = detrend(full_incidence).flatten()
#         environment_detrend = detrend(full_env).flatten()

#         dt = 1      # monthly time step
#         dj = 1 / 12
#         s0 = 2 * dt

#         try:
#             print(f"Computing wavelet coherence for {target_region} - {target_env}...")
#             WCT, aWCT, coi, freqs, sig = wavelet.wct(
#                 incidence_detrend,
#                 environment_detrend,
#                 dt=dt,
#                 dj=dj,
#                 s0=s0,
#                 wavelet=mother,
#                 normalize=False,
#                 mc_count=1000, 
#                 cache=False
#             )
#             print("Wavelet coherence computed.")
#             print(f"\n--- Properties of Wavelet Coherence Variables for {target_region} - {target_env} ---")
#             print(f"aWCT (Phase Angle) properties:")
#             print(f"  Shape: {aWCT.shape}")
#             print(f"  Data Type: {aWCT.dtype}")
#             print(f"  Min Value: {np.min(aWCT):.4f}")
#             print(f"  Max Value: {np.max(aWCT):.4f}")
#             print(f"  Mean Value: {np.mean(aWCT):.4f}")
#             print(f"  Contains NaN: {np.isnan(aWCT).any()}")
#             print(f"  Contains Inf: {np.isinf(aWCT).any()}")

#             print(f"\nWCT (Coherence) properties:")
#             print(f"  Shape: {WCT.shape}")
#             print(f"  Data Type: {WCT.dtype}")
#             print(f"  Min Value: {np.min(WCT):.4f}")
#             print(f"  Max Value: {np.max(WCT):.4f}")
#             print(f"  Mean Value: {np.mean(WCT):.4f}")
#             print(f"  Contains NaN: {np.isnan(WCT).any()}")
#             print(f"  Contains Inf: {np.isinf(WCT).any()}")

#             print(f"\ncoi (Cone of Influence) properties:")
#             print(f"  Shape: {coi.shape}")
#             print(f"  Data Type: {coi.dtype}")
#             print(f"  Min Value: {np.min(coi):.4f}")
#             print(f"  Max Value: {np.max(coi):.4f}")
#             print(f"  Mean Value: {np.mean(coi):.4f}")
#             print(f"  Contains NaN: {np.isnan(coi).any()}")
#             print(f"  Contains Inf: {np.isinf(coi).any()}")

#             print(f"\nfreqs (Frequencies) properties:")
#             print(f"  Shape: {freqs.shape}")
#             print(f"  Data Type: {freqs.dtype}")
#             print(f"  Min Value: {np.min(freqs):.4f}")
#             print(f"  Max Value: {np.max(freqs):.4f}")
#             print(f"  Mean Value: {np.mean(freqs):.4f}")
#             print(f"  Contains NaN: {np.isnan(freqs).any()}")
#             print(f"  Contains Inf: {np.isinf(freqs).any()}")
#             print(f"  Corresponding Periods (Min/Max): {1/np.max(freqs):.2f} / {1/np.min(freqs):.2f} (Months)")

#             np.set_printoptions(threshold=np.inf, suppress=True)
#             print(f"\nsig (Significance Levels) full array:\n{sig}\n")
#             print(f"\nsig (Significance Levels) properties:")
#             print(f"  Shape: {sig.shape}")
#             print(f"  Data Type: {sig.dtype}")
#             print(f"  Min Value: {np.min(sig):.4f}")
#             print(f"  Max Value: {np.max(sig):.4f}")
#             print(f"  Mean Value: {np.mean(sig):.4f}")
#             print(f"  Contains NaN: {np.isnan(sig).any()}")
#             print(f"  Contains Inf: {np.isinf(sig).any()}")
#             if np.isnan(sig).any():
#                 print("Warning: Replacing NaNs in `sig` with last valid value.")
#                 nan_mask = np.isnan(sig)
#                 last_valid = sig[~nan_mask][-1]  # or use np.nanmean(sig)
#                 sig[nan_mask] = last_valid
#             print(f"------------------------------------------------------------------\n")
#             plot_wavelet_coherence(
#                 region_df=region_df,
#                 incidence_detrend=incidence_detrend,
#                 aWCT=aWCT,
#                 WCT=WCT,
#                 coi=coi,
#                 freqs=freqs,
#                 sig=sig,
#                 incidence_col=incidence_col,
#                 target_env=target_env,
#                 target_region=target_region,
#                 SAVE_DIR=FIG_DIR,
#                 time_col=time_col
#             )
#         except Exception as e:
#             print(f"Error during wavelet coherence or plotting for {target_region}, {target_env}: {e}")

# # %%
# df = monthly_dengue_env_region.copy()
# environmental_variables = ['temperature_2m', 'temperature_2m_min',
#     'temperature_2m_max', 'precipitation',
#     'potential_evaporation_sum', 'total_evaporation_sum',
#      'ANOM1+2',  'ANOM3',  'ANOM4', 
#     'ANOM3.4', 'DMI', 'DMI_East', 'evaporative_stress_index',
#     'aridity_index']

# # Plotting
# for region in df['Region'].unique():
#     region_df = df[df['Region'] == region].sort_values(by='YearMonth')

#     for env_var in environmental_variables:
#         fig, ax1 = plt.subplots(figsize=(12, 6))
        
#         color1 = 'tab:blue'
#         ax1.set_xlabel('YearMonth')
#         ax1.set_ylabel(env_var, color=color1)
#         sns.lineplot(data=region_df, x='YearMonth', y=env_var, ax=ax1, label=env_var, color=color1, legend=False)
#         ax1.tick_params(axis='y', labelcolor=color1)

#         ax2 = ax1.twinx()
#         color2 = 'tab:red'
#         ax2.set_ylabel('Incidence Rate', color=color2)
#         sns.lineplot(data=region_df, x='YearMonth', y='Incidence_Rate', ax=ax2, label='Incidence Rate Per 100000', color=color2, legend=False)
#         ax2.tick_params(axis='y', labelcolor=color2)

#         plt.title(f'{env_var} and Incidence Rate Over Time in {region}')
#         fig.tight_layout()

#         # Combine legends from both axes
#         lines_1, labels_1 = ax1.get_legend_handles_labels()
#         lines_2, labels_2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

#         plt.grid(True)
#         plt.savefig(FIG_DIR / f'{region.replace(" ", "_")}_{env_var}_IR_timeseries.png')
#         plt.close()

# print("Time series plots generated successfully for each regional IR and environmental variable.")


# # %%
# df = monthly_dengue_env_national.copy()
# environmental_variables = ['temperature_2m', 'temperature_2m_min',
#     'temperature_2m_max', 'precipitation',
#     'potential_evaporation_sum', 'total_evaporation_sum',
#      'ANOM1+2',  'ANOM3',  'ANOM4', 
#     'ANOM3.4', 'DMI', 'DMI_East', 'evaporative_stress_index',
#     'aridity_index']

# # Plotting
# for env_var in environmental_variables:
#     fig, ax1 = plt.subplots(figsize=(12, 6))
    
#     color1 = 'tab:blue'
#     ax1.set_xlabel('YearMonth')
#     ax1.set_ylabel(env_var, color=color1)
#     sns.lineplot(data=df, x='YearMonth', y=env_var, ax=ax1, label=env_var, color=color1, legend=False)
#     ax1.tick_params(axis='y', labelcolor=color1)
    
#     ax2 = ax1.twinx()  # Create second y-axis sharing the same x-axis
    
#     color2 = 'tab:red'
#     ax2.set_ylabel('Incidence Rate', color=color2)
#     sns.lineplot(data=df, x='YearMonth', y='Incidence_Rate', ax=ax2, label='Incidence Rate Per 100000', color=color2, legend=False)
#     ax2.tick_params(axis='y', labelcolor=color2)
    
#     plt.title(f'{env_var} and Incidence Rate Over Time in Indonesia')
#     fig.tight_layout()
    
#     # Combine legends from both axes
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
#     plt.grid(True)
#     plt.savefig(FIG_DIR / f'Indonesia_{env_var}_IR_timeseries.png')
#     plt.close()

# print("Time series plots generated successfully for environmental variable and IR pairs at national level.")


# %%



