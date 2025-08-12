import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBRegressor
import shap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import PartialDependenceDisplay
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import cudf
import cupy as cp
import optuna

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KT0812"

# --- Run sweep for the entire nation ---
n_trials_for_national_study = 50 # Number of trials for the national study

# Normalize and validate input
config_input = config_input.strip().lower()
if len(config_input) != 6:
    raise ValueError("Invalid config format. Use 6 characters like 'kt0726' or 'wf2031'.")

# Parse components
validation_flag = config_input[0]
landuse_flag = config_input[1]
version_digits = config_input[2:]

# Determine validation strategy
if validation_flag == 'k':
    validation_strategy = 'kfold'
elif validation_flag == 'w':
    validation_strategy = 'walk'
else:
    raise ValueError("Invalid validation flag. Use 'K' for kfold or 'W' for walk_forward.")

# Determine land use flag
if landuse_flag == 't':
    USE_LANDUSE_FEATURES = True
    landuse_suffix = "lulc"
elif landuse_flag == 'f':
    USE_LANDUSE_FEATURES = False
    landuse_suffix = ""
else:
    raise ValueError("Land use flag must be 'T' or 'F'.")

# Validate version digits
if not version_digits.isdigit():
    raise ValueError("Version must be 4 digits.")
VERSION_STAMP = version_digits
version_suffix = f"{VERSION_STAMP}"

# --- Logging ---
print(f"Using validation strategy: {validation_strategy} in version {VERSION_STAMP}")
print(f"Land Use Features Included: {USE_LANDUSE_FEATURES}")

# Define project root based on notebook location (assuming this part is correct for your setup)
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve() # fallback

PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = REPORTS_DIR / "tables"

# Load the data once outside the train function for efficiency
df = pd.read_csv(PROCESSED_DIR / "monthly_dengue_env_id_updated.csv")

# --- REGION REASSIGNMENT (Keep this for consistency) ---
df['Region_Group'] = df['Region'].replace({'Maluku Islands': 'Maluku-Papua', 'Papua': 'Maluku-Papua'})
print("--- DataFrame after Region_Group creation ---")
print(df['Region_Group'].value_counts())
print("-" * 50)

df['YearMonth'] = pd.to_datetime(df['YearMonth']) # Ensure YearMonth is datetime

# Define variable categories
env_vars = [
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
    'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum',
    'evaporative_stress_index', 'aridity_index',
    'temperature_2m_ANOM', 'temperature_2m_min_ANOM', 'temperature_2m_max_ANOM',
    'potential_evaporation_sum_ANOM', 'total_evaporation_sum_ANOM', 'precipitation_ANOM'
]

land_use_vars = [
    'Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95',
    'Class_30', 'Class_20', 'Class_10', 'Class_90', 'Class_80'
]

climate_vars = ['ANOM1+2', 'ANOM3', 'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East']
target = 'Incidence_Rate'  

# Sort data by time and region
df = df.sort_values(['YearMonth', 'ID_2'])

# Create lag features for environmental and climate variables
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

# Compile feature list
features = []
for var in env_vars + climate_vars:
    if var in df.columns:
        features.append(var)
if USE_LANDUSE_FEATURES:
    for var in land_use_vars:
        if var in df.columns:
            features.append(var)
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            lagged_var = f'{var}_lag{lag}'
            if lagged_var in df.columns:
                features.append(lagged_var)

# Final feature list excluding metadata and target
actual_feature_columns = [
    col for col in features
    if col not in ['YearMonth', 'ID_2', 'Year', target]
]

print("\n--- Final list of features for the model ---")
print(actual_feature_columns)
print(f"Total features: {len(actual_feature_columns)}")
print("-" * 50)
print("--- Creating splits based on validation strategy ---")

# Data for hyperparameter tuning
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy().dropna(subset=actual_feature_columns + [target])
# Data for final, unseen test
df_test_national = df[df['YearMonth'].dt.year == 2023].copy().dropna(subset=actual_feature_columns + [target])

print(f"Shape of df_train_val for National: {df_train_val_national.shape}")
print(f"Shape of df_final_test for National: {df_test_national.shape}")

# Convert the full processed DataFrame to cudf and cupy once
X_gpu = cudf.DataFrame(df_train_val_national[actual_feature_columns])
y_gpu = cudf.DataFrame(df_train_val_national[[target]])

splits = []

if validation_strategy == 'kfold':
    n_splits = 5
    # KFold works with pandas indices
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=64)
    for train_index, test_index in kf.split(df_train_val_national):
        # Store the indices as cupy arrays
        splits.append((cp.array(train_index), cp.array(test_index)))

elif validation_strategy == 'walk':
    initial_train_months = 60
    test_window = 6
    unique_time_periods = df_train_val_national['YearMonth'].unique()
    n_time_periods = len(unique_time_periods)
    n_walks = (n_time_periods - initial_train_months) // test_window

    for i in range(n_walks):
        train_end_period_idx = initial_train_months + i * test_window
        test_start_period_idx = train_end_period_idx
        test_end_period_idx = test_start_period_idx + test_window

        if test_end_period_idx > n_time_periods:
            break

        train_end_time = unique_time_periods[train_end_period_idx - 1]
        test_start_time = unique_time_periods[test_start_period_idx]
        test_end_time = unique_time_periods[test_end_period_idx - 1]

        # Get pandas indices for the time periods
        train_indices_pd = df_train_val_national.loc[df_train_val_national['YearMonth'] <= train_end_time].index
        test_indices_pd = df_train_val_national.loc[
            (df_train_val_national['YearMonth'] >= test_start_time) & (df_train_val_national['YearMonth'] <= test_end_time)
        ].index
        
        # Convert indices to cupy arrays
        splits.append((cp.array(train_indices_pd.to_numpy()), cp.array(test_indices_pd.to_numpy())))

print(f"Pre-calculated {len(splits)} splits for {validation_strategy} validation.")

def objective(trial, X_gpu, y_gpu, splits):
    """
    Objective function for Optuna to minimize.
    It performs cross-validation or walk-forward validation and returns the
    overall RMSE.
    """
    # --- Suggest Hyperparameters to Optuna ---
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 64,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'subsample': trial.suggest_float('subsample', 0.1, 0.5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
        'gamma': trial.suggest_float('gamma', 0.1, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 100, log=True),
    }
    num_boost_round = trial.suggest_int('n_estimators', 50, 7000)

    all_preds = []
    all_true = []

    for train_index_gpu, test_index_gpu in splits:
        X_train_fold_gpu = X_gpu.loc[train_index_gpu]
        y_train_fold_gpu = y_gpu.loc[train_index_gpu]
        X_test_fold_gpu = X_gpu.loc[test_index_gpu]
        y_test_fold_gpu = y_gpu.loc[test_index_gpu]

        if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
            continue

        # Prepare DMatrices
        dtrain = xgboost.DMatrix(X_train_fold_gpu, label=y_train_fold_gpu)
        dtest = xgboost.DMatrix(X_test_fold_gpu, label=y_test_fold_gpu)

        # Train using xgboost.train
        booster = xgboost.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, 'test')],
            verbose_eval=False
        )

        # Predict
        predictions_gpu = booster.predict(dtest)

        all_preds.append(predictions_gpu)
        all_true.append(y_test_fold_gpu.to_numpy().flatten())

    if all_true:
        global_overall_rmse = np.sqrt(mean_squared_error(
            np.concatenate(all_true),
            np.concatenate(all_preds)
        ))
        print(f"Trial finished with RMSE: {global_overall_rmse:.2f}")
        return global_overall_rmse
    else:
        return float('inf')


if df_train_val_national.empty:
    print(f"No training/validation data for national study. Cannot proceed.")
else:
    # --- Optuna Study Setup (offline-friendly) ---
    # The storage argument saves the study to a local SQLite database
    study_name = f"-xgbR-nation-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True
    )
    
    print(f"Optuna study created/loaded: {study_name}.db")
    print(f"Starting {n_trials_for_national_study} trials...")
    
    # Run the optimization
    study.optimize(
    lambda trial: objective(
        trial,
        X_gpu=X_gpu,
        y_gpu=y_gpu,
        splits=splits
    ),
    n_trials=n_trials_for_national_study,
    n_jobs=-1
)

    print("\nNational study completed.")
    
    # --- Retrieve and Log Best Hyperparameters for the Nation ---
    print("\n--- Best Hyperparameters Found by Optuna ---")
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best RMSE: {best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Prepare data for CSV saving
    all_best_hypers = [{
        'Region': 'National',
        'best_rmse': best_value,
        **best_params
    }]

    # --- Final Aggregation and CSV Saving ---
    if all_best_hypers:
        best_hypers_df = pd.DataFrame(all_best_hypers)
        best_hypers_csv_path = TABLES_DIR / f"{study_name}_params.csv"
        best_hypers_df.to_csv(best_hypers_csv_path, index=False)
        print(f"\nSaved best national hyperparameters to {best_hypers_csv_path}")
    else:
        print("\nNo best national hyperparameters found to save.")

print("\n--- NATIONAL Study and Hyperparameter Retrieval Completed ---")

# --- Test with Final Model (This section remains largely the same) ---
print("\n--- Begin Final Model Training and Evaluation ---")

# Read hyperparameters for the national model
best_hypers_csv_path = TABLES_DIR / f"{study_name}_params.csv"
hyperparams_df = pd.read_csv(best_hypers_csv_path)
print(f"Extracted hyperparameters for national model: {best_hypers_csv_path}")

# Extract hyperparameters for the national model
params = hyperparams_df.iloc[0]  # Assuming the first row contains national hyperparameters
national_hyperparams = {
    'gamma': params['gamma'],
    'n_estimators': int(params['n_estimators']),
    'max_depth': int(params['max_depth']),
    'reg_alpha': params['reg_alpha'],
    'subsample': params['subsample'],
    'reg_lambda': params['reg_lambda'],
    'learning_rate': params['learning_rate'],
    'colsample_bytree': params['colsample_bytree'],
    'min_child_weight': int(params['min_child_weight'])
}

# Convert pand

all_shap_plots = []
all_pdp_plots = []  
national_summary_data = {}

# Convert pandas DataFrames to cuDF DataFrames for GPU acceleration
X_train = cudf.DataFrame(df_train_val_national[actual_feature_columns])
y_train = cudf.DataFrame(df_train_val_national[[target]])
X_test = cudf.DataFrame(df_test_national[actual_feature_columns])
y_test = cudf.DataFrame(df_test_national[[target]])
X_full = cudf.DataFrame(df[actual_feature_columns])
y_full = cudf.DataFrame(df[[target]])

# Select a subset of X_train for SHAP background sampling
background_sample = df_train_val_national[actual_feature_columns].sample(n=100, random_state=42)

# Initialize and train the XGBoost Regressor model
model = xgboost.XGBRegressor(
    objective='reg:squarederror', # Objective for regression tasks
    tree_method='hist',           # Use histogram-based tree method for faster training
    device='cuda',                # Specify GPU device for training
    random_state=64,              # For reproducibility
    **national_hyperparams        # Unpack national hyperparameters
)
model.fit(X_train, y_train)

# Initialize dictionary to store metrics for the national model
current_national_metrics = {}

for label, X_subset, y_subset in [
    ("Train/Val", X_train, y_train),
    ("Test", X_test, y_test),
    ("Full", X_full, y_full)
]:
    preds = model.predict(X_subset)
    preds = np.maximum(0, cp.asnumpy(preds))  # CuPy -> NumPy
    y_true = y_subset.to_numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    current_national_metrics[f'{label} RMSE'] = rmse
    current_national_metrics[f'{label} MAE'] = mae
    current_national_metrics[f'{label} R2'] = r2

    # Prepare values for SHAP tree explainer -> 
    # takes strictly numpy arrays or pandas DataFrames
    X_pd = X_subset.to_pandas()
    explainer = shap.TreeExplainer(model, data=background_sample, feature_perturbation="auto")
    shap_vals = explainer.shap_values(X_pd.values)
    shap_vals_np = cp.asnumpy(shap_vals)   # For plotting

    all_shap_plots.append((shap_vals_np, X_pd, f"National - {label}", rmse, mae, r2))

    if label == "Test":
        mean_abs_shap = np.abs(shap_vals_np).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=X_pd.columns)
        top_5_features = feature_importance.nlargest(5).index.tolist()
        print(f"Top 5 predictors for National (Test Set): {top_5_features}")

        for feature in top_5_features:
            all_pdp_plots.append((model, X_pd, feature, "National"))

# Store all calculated metrics and Log Incidence Rate statistics for the national model
national_summary_data['National'] = current_national_metrics
national_summary_data['National']['Log_IR Min'] = df[target].min()
national_summary_data['National']['Log_IR Max'] = df[target].max()
national_summary_data['National']['Log_IR 25th Quantile'] = df[target].quantile(0.25)
national_summary_data['National']['Log_IR 50th Quantile'] = df[target].quantile(0.50)
national_summary_data['National']['Log_IR 75th Quantile'] = df[target].quantile(0.75)

# Generate SHAP plots and save them to a PDF file
pdf_output_filename = f"xgb_national_{validation_strategy}_plots_{landuse_suffix}_{version_digits}.pdf"
pdf_path = FIGURES_DIR / pdf_output_filename

with PdfPages(pdf_path) as pdf:
    # SHAP summary plots
    for shap_vals, X_pd, title, rmse, mae, r2 in all_shap_plots:
        fig, ax = plt.subplots(figsize=(10, 8 + len(actual_feature_columns) * 0.25))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_pd, show=False)
        ax.set_title(f"{title} | RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    # PDP plots using sklearn
    for model, X_pd, feature, title in all_pdp_plots:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X_pd,
                [feature],
                ax=ax,
                kind='average',
                grid_resolution=100
            )
            ax.set_title(f'{title} - Sklearn PDP - {feature}')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Failed sklearn PDP for {title} - {feature}: {e}")


print(f"National SHAP and PDP plots saved to '{pdf_path}'")

# --- Generate National Summary Table ---
print("\n--- National Summary Table ---")

# Convert the collected summary data into a pandas DataFrame
summary_df = pd.DataFrame.from_dict(national_summary_data, orient='index').T

# Save the DataFrame to a CSV file
csv_output_filename = f"xgb_national_{validation_strategy}_test_{landuse_suffix}_{version_digits}.csv"
csv_filename = TABLES_DIR / csv_output_filename
summary_df.to_csv(csv_filename, index=True, float_format="%.2f")

print(f"National summary table saved to '{csv_filename}'")
print("-" * 50)