import argparse
import os
import pathlib
import sys
import toml
import numpy as np
import polars as pl
from loguru import logger
import random
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# --- 设置相对路径导入 (If needed, assumes script is run from project root or similar) ---
# script_dir = pathlib.Path(__file__).parent.resolve()
# project_root = script_dir.parent.parent # Adjust as needed
# sys.path.insert(0, str(project_root))
# try:
#     pass # Import custom modules if necessary
# except ImportError as e:
#     logger.error(f"Failed to import custom modules: {e}")
#     sys.exit(1)

# --- 日志和精度配置 ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- Helper Functions ---
def load_config() -> dict:
    """Loads the entire config.toml file from the script's parent directory."""
    script_dir = pathlib.Path(__file__).parent.resolve()
    project_root = script_dir.parent
    config_path = project_root / "config.toml"
    config = {}
    if config_path.is_file():
        logger.info(f"Loading configuration from: {config_path}")
        try:
            config = toml.load(config_path)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.exception(f"Error loading config file {config_path}: {e}")
            config = {} # Return empty dict on error
    else:
        logger.warning(f"Configuration file not found at {config_path}. Relying on defaults or CLI args.")
    return config

def find_optimal_threshold(y_true, y_prob, beta=1.2) -> Tuple[float, float, float, float]:
    """Finds the optimal threshold maximizing G-mean biased towards recall."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Add small epsilon to avoid division by zero or log(0) if tpr or (1-fpr) is 0
    epsilon = 1e-9 
    gmeans = np.sqrt(np.maximum(tpr, epsilon)**beta * np.maximum(1-fpr, epsilon)) 
    try:
        ix = np.nanargmax(gmeans) # Use nanargmax to handle potential NaNs
        threshold = thresholds[ix]
        roc_auc = auc(fpr, tpr)
        return threshold, fpr[ix], tpr[ix], roc_auc
    except (ValueError, IndexError) as e:
         logger.warning(f"Could not determine optimal threshold: {e}. Returning default 0.5")
         # Calculate AUC anyway if possible
         roc_auc = auc(fpr, tpr) if len(fpr) > 1 and len(tpr) > 1 else 0.0
         return 0.5, np.nan, np.nan, roc_auc # Default threshold


def load_and_combine_preference_data(pref_dir: pathlib.Path) -> Optional[pl.DataFrame]:
    """Recursively loads CSVs from pref_dir, finds matching NPYs, and combines data."""
    all_pref_dfs = []
    csv_files = list(pref_dir.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} potential CSV files in {pref_dir}.")

    if not csv_files:
        logger.error(f"No CSV files found in preference directory: {pref_dir}")
        return None

    loaded_ids = set() # Keep track of loaded IDs to avoid duplicates across files

    for csv_path in csv_files:
        npy_path = csv_path.with_suffix('.npy')
        logger.debug(f"Processing CSV: {csv_path}")
        if not npy_path.is_file():
            logger.warning(f"Matching NPY file not found for {csv_path}. Skipping this file.")
            continue

        try:
            df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
            logger.debug(f"Loaded CSV {csv_path} with shape {df_csv.shape}")
            required_cols = {'id', 'preference'}
            if not required_cols.issubset(df_csv.columns):
                logger.warning(f"CSV {csv_path} missing required columns ({required_cols}). Skipping.")
                continue

            embeddings = np.load(npy_path)
            logger.debug(f"Loaded NPY {npy_path} with shape {embeddings.shape}")

            if df_csv.height != embeddings.shape[0]:
                logger.warning(f"Row count mismatch between {csv_path} ({df_csv.height}) and {npy_path} ({embeddings.shape[0]}). Skipping.")
                continue
                
            # Filter for valid preferences and map to label
            valid_prefs = {'like', 'dislike'}
            df_csv = df_csv.filter(pl.col('preference').is_in(valid_prefs))
            if df_csv.is_empty():
                 logger.warning(f"No valid 'like' or 'dislike' preferences found in {csv_path}. Skipping.")
                 continue

            df_csv = df_csv.with_columns(
                pl.when(pl.col('preference') == 'like').then(1).otherwise(0).alias('label')
            )
            
            # Add embeddings as a list column
            # Convert numpy array rows to lists for Polars List type
            embedding_lists = [row.tolist() for row in embeddings] 
            df_csv = df_csv.with_columns(pl.Series("embedding", embedding_lists))

            # Filter out already loaded IDs (keep first occurrence)
            current_ids = set(df_csv['id'].to_list())
            new_ids_mask = df_csv['id'].is_in(loaded_ids).not_()
            df_new = df_csv.filter(new_ids_mask)
            
            if not df_new.is_empty():
                 all_pref_dfs.append(df_new.select(["id", "preference", "label", "embedding"]))
                 newly_added_ids = set(df_new['id'].to_list())
                 loaded_ids.update(newly_added_ids)
                 logger.info(f"Added {df_new.height} new unique entries from {csv_path}.")
            else:
                 logger.info(f"No new unique entries found in {csv_path}.")


        except Exception as e:
            logger.exception(f"Error processing file pair: {csv_path} / {npy_path}")

    if not all_pref_dfs:
        logger.error("Could not load any valid preference data.")
        return None

    df_combined_prefs = pl.concat(all_pref_dfs, how="vertical_relaxed") # Use relaxed in case schema differs slightly (though shouldn't)
    logger.info(f"Combined preference data shape: {df_combined_prefs.shape}")
    return df_combined_prefs

def load_background_data(bg_csv_path_str: str) -> Optional[pl.DataFrame]:
    """Loads background CSV and checks for corresponding NPY file."""
    bg_csv_path = pathlib.Path(bg_csv_path_str)
    npy_path = bg_csv_path.with_suffix(".npy")

    if not bg_csv_path.is_file():
         logger.error(f"Background CSV file not found: {bg_csv_path}")
         return None
    if not npy_path.is_file():
        logger.error(f"Corresponding background NPY file not found: {npy_path}")
        return None

    try:
        df_csv = pl.read_csv(bg_csv_path, schema_overrides={'id': pl.Utf8})
        embeddings = np.load(npy_path)

        if df_csv.height != embeddings.shape[0]:
            logger.error(f"Row count mismatch for background data: {bg_csv_path} vs {npy_path}")
            return None
        
        if 'id' not in df_csv.columns:
            logger.error(f"Background CSV {bg_csv_path} missing 'id' column.")
            return None

        embedding_lists = [row.tolist() for row in embeddings]
        df_csv = df_csv.with_columns(pl.Series("embedding", embedding_lists))
        logger.info(f"Loaded background data ({bg_csv_path.name}) shape: {df_csv.shape}")
        return df_csv.select(["id", "embedding"]) # Keep only necessary columns

    except Exception as e:
        logger.exception(f"Error loading background data from {bg_csv_path}")
        return None

def load_target_data(target_csv_path_str: str) -> Optional[pl.DataFrame]:
    """Loads target CSV and checks for corresponding NPY file."""
    target_csv_path = pathlib.Path(target_csv_path_str)
    npy_path = target_csv_path.with_suffix(".npy")

    if not target_csv_path.is_file():
        logger.error(f"Target CSV file not found: {target_csv_path}")
        return None
    if not npy_path.is_file():
        logger.error(f"Corresponding target NPY file not found: {npy_path}")
        return None
    try:
        # Load all columns from target CSV
        df_csv = pl.read_csv(target_csv_path, schema_overrides={'id': pl.Utf8})
        embeddings = np.load(npy_path)

        if df_csv.height != embeddings.shape[0]:
            logger.error(f"Row count mismatch for target data: {target_csv_path} vs {npy_path}")
            return None
            
        if 'id' not in df_csv.columns:
            logger.error(f"Target CSV {target_csv_path} missing 'id' column.")
            return None

        embedding_lists = [row.tolist() for row in embeddings]
        df_csv = df_csv.with_columns(pl.Series("embedding", embedding_lists))
        logger.info(f"Loaded target data ({target_csv_path.name}) shape: {df_csv.shape}")
        return df_csv
    except Exception as e:
        logger.exception(f"Error loading target data from {target_csv_path}")
        return None


def prepare_training_data(df_prefs: pl.DataFrame, df_bg: pl.DataFrame, neg_ratio: float, random_state: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Prepares training data by sampling negatives from background."""
    
    logger.info("Preparing training data with negative sampling...")
    
    # Separate positive and negative preferences
    df_pos = df_prefs.filter(pl.col('label') == 1)
    df_neg_explicit = df_prefs.filter(pl.col('label') == 0)
    
    n_positive = df_pos.height
    n_neg_explicit = df_neg_explicit.height
    
    if n_positive == 0:
        logger.error("No positive preference samples found. Cannot train.")
        return None

    # Remove preference IDs from background pool
    pref_ids = set(df_prefs['id'].to_list())
    df_bg_pool = df_bg.filter(pl.col('id').is_in(pref_ids).not_())
    n_available_bg = df_bg_pool.height
    
    if n_available_bg == 0:
        logger.warning("No background samples available after removing preference IDs.")
        n_to_sample = 0
    else:
        # Calculate number of negatives to sample
        n_target_negative = int(n_positive * neg_ratio)
        n_to_sample = max(0, n_target_negative - n_neg_explicit)
        n_to_sample = min(n_to_sample, n_available_bg) # Can't sample more than available
        logger.info(f"Target negative count: {n_target_negative}. Explicit negatives: {n_neg_explicit}. Need to sample: {n_to_sample} from {n_available_bg} background samples.")

    # Sample negatives
    if n_to_sample > 0:
        df_neg_sampled = df_bg_pool.sample(n=n_to_sample, shuffle=True, seed=random_state)
        df_neg_sampled = df_neg_sampled.with_columns(pl.lit(0).alias('label'))
        # Add preference column with null/default value for schema consistency if needed later? No, only need id, embedding, label
        df_neg_sampled = df_neg_sampled.select(["id", "embedding", "label"]) 
    else:
        df_neg_sampled = pl.DataFrame({"id": [], "embedding": [], "label": []}, schema={"id": pl.Utf8, "embedding": pl.List(pl.Float32), "label": pl.Int8}) # Empty DF with schema

    # Combine all data
    # Need common columns: id, embedding, label
    df_train = pl.concat([
        df_pos.select(["id", "embedding", "label"]), 
        df_neg_explicit.select(["id", "embedding", "label"]), 
        df_neg_sampled
        ], how="vertical_relaxed")

    # Final checks and conversion to NumPy
    if df_train.is_empty():
        logger.error("Training data is empty after processing.")
        return None
        
    final_counts = df_train['label'].value_counts()
    pos_count = final_counts.filter(pl.col('label') == 1).select('count').item(0,0) if 1 in final_counts['label'] else 0
    neg_count = final_counts.filter(pl.col('label') == 0).select('count').item(0,0) if 0 in final_counts['label'] else 0
    
    if pos_count == 0 or neg_count == 0:
         logger.error(f"Training data lacks samples for both classes (Pos: {pos_count}, Neg: {neg_count}). Cannot train.")
         return None
         
    logger.info(f"Final training data distribution - Positive (like): {pos_count}, Negative (dislike+sampled): {neg_count}")
    
    try:
        # Ensure embeddings are not null before converting
        df_train = df_train.filter(pl.col('embedding').is_not_null())
        if df_train.is_empty():
             logger.error("Training data is empty after filtering null embeddings.")
             return None
             
        X = np.array(df_train['embedding'].to_list(), dtype=np.float64) # Use float64
        y = df_train['label'].to_numpy()

        # Check for NaN/Inf in original features X
        if np.isnan(X).any(): # Check original X
            nan_rows = np.isnan(X).any(axis=1)
            logger.error(f"NaN values found in {np.sum(nan_rows)} rows of the training features (X). First 5 problematic IDs: {df_train.filter(pl.lit(nan_rows))['id'].head(5).to_list()}")
            return None
        if np.isinf(X).any():
            inf_rows = np.isinf(X).any(axis=1)
            logger.error(f"Infinite values found in {np.sum(inf_rows)} rows of the training features (X). First 5 problematic IDs: {df_train.filter(pl.lit(inf_rows))['id'].head(5).to_list()}")
            return None

        logger.info(f"Prepared training features X shape: {X.shape}, target y shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.exception("Error converting training data to NumPy arrays.")
        return None


def perform_cv_and_get_threshold(X: np.ndarray, y: np.ndarray, n_splits: int, beta: float, random_state: int, C: float = 1.0, max_iter: int = 1000) -> Optional[float]:
    """Performs K-Fold CV, logs metrics, returns overall optimal threshold."""
    logger.info(f"Starting {n_splits}-fold cross-validation (beta={beta}, C={C}, max_iter={max_iter})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_true = []
    all_probs = []
    fold_aucs = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate model with specified C, max_iter, tol and saga solver
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)#, class_weight='balanced')
        try:
            model.fit(X_train, y_train) # Fit on original data
            y_prob = model.predict_proba(X_test)[:, 1] # Predict on original data
        except ValueError as ve:
            logger.error(f"ValueError during model fitting/prediction in fold {fold_idx+1}: {ve}")
            # Log info about the original data if error occurs
            logger.error(f"X_train shape: {X_train.shape}, contains NaN: {np.isnan(X_train).any()}, contains Inf: {np.isinf(X_train).any()}")
            logger.error(f"X_test shape: {X_test.shape}, contains NaN: {np.isnan(X_test).any()}, contains Inf: {np.isinf(X_test).any()}")
            # Skip this fold or handle error appropriately
            logger.warning(f"Skipping fold {fold_idx+1} due to fitting error.")
            continue # Skip to the next fold

        # Calculate metrics for the fold
        fold_threshold, fold_fpr, fold_tpr, fold_auc = find_optimal_threshold(y_test, y_prob, beta)
        fold_aucs.append(fold_auc)
        all_true.extend(y_test)
        all_probs.extend(y_prob)
        logger.info(f"Fold {fold_idx+1}/{n_splits} AUC: {fold_auc:.4f}, Threshold: {fold_threshold:.4f}")

    if not all_probs:
         logger.error("Cross-validation did not produce any probability predictions.")
         return None

    overall_threshold, _, _, overall_auc = find_optimal_threshold(all_true, all_probs, beta)
    logger.info(f"--- CV Summary ---")
    logger.info(f"Average Fold AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    logger.info(f"Overall AUC based on all folds: {overall_auc:.4f}")
    logger.info(f"Optimal Threshold based on all folds (beta={beta}): {overall_threshold:.4f}")
    
    # Print overall classification report for CV predictions
    try:
        preds_overall = (np.array(all_probs) >= overall_threshold).astype(int)
        report = classification_report(all_true, preds_overall, target_names=['dislike/neg', 'like/pos'], zero_division=0)
        logger.info("--- Overall CV Classification Report (using overall threshold) ---")
        for line in report.split('\n'):
            logger.info(line)
    except Exception as e:
        logger.warning(f"Could not generate overall CV classification report: {e}")

    return overall_threshold


def biased_sample(scores: np.ndarray, threshold: float, vicinity_margin: float = 0.1) -> np.ndarray:
    """Performs biased sampling based on scores and threshold."""
    logger.info(f"Performing biased sampling with threshold={threshold:.4f}, vicinity_margin={vicinity_margin:.2f}")
    n_samples = len(scores)
    show = np.zeros(n_samples, dtype=bool)
    
    # 1. Select all high-probability samples
    high_prob_threshold = threshold + vicinity_margin
    high_prob_indices = np.where(scores >= high_prob_threshold)[0]
    show[high_prob_indices] = True
    logger.info(f"Selected {len(high_prob_indices)} high-probability samples (score >= {high_prob_threshold:.4f})")

    # 2. Probabilistically sample from the vicinity
    lower_bound = max(0.0, threshold - vicinity_margin) # Ensure lower bound is not negative
    upper_bound = high_prob_threshold
    vicinity_indices = np.where((scores >= lower_bound) & (scores < upper_bound))[0]
    
    sampled_in_vicinity = 0
    if len(vicinity_indices) > 0:
        vicinity_scores = scores[vicinity_indices]
        # Scale probability linearly from 0 at lower_bound to 1 at upper_bound
        # Handle edge case where lower_bound equals upper_bound (should not happen with margin > 0)
        prob_range = upper_bound - lower_bound
        if prob_range <= 0:
            logger.warning("Sampling probability range is zero or negative. Setting vicinity probabilities to 0.5")
            probabilities = np.full_like(vicinity_scores, 0.5)
        else:
            probabilities = (vicinity_scores - lower_bound) / prob_range
            probabilities = np.clip(probabilities, 0.0, 1.0) # Ensure probabilities are valid

        random_values = np.random.rand(len(vicinity_indices))
        sampled_mask = random_values < probabilities
        show[vicinity_indices[sampled_mask]] = True
        sampled_in_vicinity = np.sum(sampled_mask)
        logger.info(f"Sampled {sampled_in_vicinity} items from the threshold vicinity [{lower_bound:.4f}, {upper_bound:.4f}) probabilistically.")
    else:
         logger.info("No items found in the threshold vicinity for probabilistic sampling.")
         
    total_shown = np.sum(show)
    logger.info(f"Total items marked to show: {total_shown} / {n_samples}")
    return show.astype(np.uint8) # Return as 0/1


# --- Main Script Logic ---
def main():
    # Load the entire config
    full_config = load_config()
    # Get specific sections, defaulting to empty dict if section is missing
    model_fitting_cfg = full_config.get('model_fitting', {})

    parser = argparse.ArgumentParser(description="Train preference model, predict scores, and sample. Reads defaults from config.toml [model_fitting].")
    # Updated arguments to read defaults from model_fitting_cfg
    parser.add_argument("--preference-dir", "-p", type=str, 
                        default=model_fitting_cfg.get('preference_dir'),
                        help="Directory containing preference CSV+NPY files.")
    parser.add_argument("--background-file", "-b", type=str, 
                        default=model_fitting_cfg.get('background_file'),
                        help="Path to the background data CSV file (expects corresponding .npy file).")
    parser.add_argument("--target-file", "-t", type=str, 
                        default=model_fitting_cfg.get('target_file'),
                        help="Path to the target prediction CSV file (expects corresponding .npy file).")
    parser.add_argument("--neg-ratio", type=float, 
                        default=model_fitting_cfg.get('neg_ratio', 5.0),
                        help="Target ratio of negative to positive samples after sampling.")
    parser.add_argument("--folds", "-k", type=int, 
                        default=model_fitting_cfg.get('folds', 5),
                        help="Number of folds for cross-validation.")
    parser.add_argument("--beta", type=float, 
                        default=model_fitting_cfg.get('beta', 1.2),
                        help="Beta value for F-beta/G-mean optimization in threshold finding.")
    # Removed output-model argument
    # Removed output-prediction-csv argument
    parser.add_argument("--random-state", type=int, 
                        default=model_fitting_cfg.get('random_state', 42),
                        help="Random state for reproducibility.")

    args = parser.parse_args()

    # --- Set Random Seeds for Reproducibility ---
    seed = args.random_state
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seeds (python, numpy) to: {seed}")

    # --- Load Data --- 
    logger.info("--- Loading Data ---")
    # Check required arguments after parsing (considering defaults from config)
    pref_dir_str = args.preference_dir
    bg_csv_path_str = args.background_file # Renamed for clarity
    target_csv_path_str = args.target_file   # Renamed for clarity

    missing_required = []
    if not pref_dir_str:
        missing_required.append('--preference-dir/-p')
    if not bg_csv_path_str:
        missing_required.append('--background-file/-b')
    if not target_csv_path_str:
         missing_required.append('--target-file/-t')
         
    if missing_required:
         logger.error(f"Missing required arguments: {', '.join(missing_required)}. Provide via command line or config.toml [model_fitting].")
         parser.print_help()
         sys.exit(1)

    pref_dir_path = pathlib.Path(pref_dir_str)
    # bg_base_path = pathlib.Path(bg_file_str)
    # target_base_path = pathlib.Path(target_file_str)

    df_prefs = load_and_combine_preference_data(pref_dir_path)
    if df_prefs is None:
        sys.exit(1)

    # Pass the full CSV path string to the loading function
    df_bg = load_background_data(bg_csv_path_str)
    if df_bg is None:
        logger.error("Failed to load background data. Cannot proceed with negative sampling.")
        sys.exit(1)

    # Pass the full CSV path string to the loading function
    df_target = load_target_data(target_csv_path_str)
    if df_target is None:
        logger.error("Failed to load target data for prediction.")
        sys.exit(1)
        
    # Ensure target has embeddings
    if 'embedding' not in df_target.columns or df_target['embedding'].is_null().any():
         logger.error("Target data is missing embeddings or contains null embeddings. Cannot predict.")
         sys.exit(1)


    # --- Prepare Training Data ---
    logger.info("--- Preparing Training Data ---")
    training_data = prepare_training_data(df_prefs, df_bg, args.neg_ratio, args.random_state)
    if training_data is None:
        sys.exit(1)
    X_train, y_train = training_data
    
    if X_train.shape[0] < args.folds:
         logger.warning(f"Insufficient training samples ({X_train.shape[0]}) for {args.folds}-fold CV. Reducing folds to {X_train.shape[0]}.")
         args.folds = X_train.shape[0] # Adjust folds if needed
         if args.folds < 2:
              logger.error("Less than 2 samples for CV. Cannot proceed.")
              sys.exit(1)

    # --- Cross-validation ---
    logger.info("--- Cross-validation --- ")
    # Define CV parameters (consider making C and max_iter CLI args later if needed)
    cv_C = 0.1
    cv_max_iter = 5000
    optimal_threshold = perform_cv_and_get_threshold(
        X_train, y_train, args.folds, args.beta, args.random_state, C=cv_C, max_iter=cv_max_iter
    )
    if optimal_threshold is None:
        logger.error("Failed to determine optimal threshold from CV. Exiting.")
        sys.exit(1)

    # --- Final Model Training ---
    logger.info("--- Training Final Model --- ")
    # Use the same C, max_iter, tol and solver for the final model
    final_model = LogisticRegression(C=cv_C, max_iter=cv_max_iter, random_state=args.random_state)#, class_weight='balanced')
    try:
        final_model.fit(X_train, y_train) # Train on original data
        logger.info("Final model trained successfully.")
    except Exception as e:
        logger.exception("Error training final model.")
        sys.exit(1)

    # --- Prediction on Target Data ---
    logger.info("--- Predicting on Target Data ---")
    try:
        X_target = np.array(df_target['embedding'].to_list(), dtype=np.float32) # Reverted to float32

        # Check for NaN/Inf in target features X_target
        logger.info("Checking target data for NaN/Inf.") # Added log
        if np.isnan(X_target).any():
            logger.error("NaN values found in target features (X_target). Cannot predict accurately.")
            # Optionally, handle or exit
            sys.exit(1) # Exit if target data is problematic
        if np.isinf(X_target).any():
            logger.error("Infinite values found in target features (X_target). Cannot predict accurately.")
            sys.exit(1)

        target_scores = final_model.predict_proba(X_target)[:, 1]
        df_target = df_target.with_columns(pl.Series("score", target_scores))
        logger.info("Prediction scores calculated for target data.")
    except Exception as e:
        logger.exception("Error predicting on target data.")
        sys.exit(1)

    # --- Biased Sampling ---
    logger.info("--- Performing Biased Sampling ---")
    show_flags = biased_sample(target_scores, optimal_threshold) # Uses default margin
    df_target = df_target.with_columns(pl.Series("show", show_flags))

    # --- Save Output ---
    logger.info("--- Saving Results ---")
    # Determine the output path - always overwrite the target CSV
    # Use the input target CSV path string directly
    output_pred_path = pathlib.Path(target_csv_path_str)
    logger.warning(f"Output CSV path set to overwrite target file: {output_pred_path}")
        
    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Drop embedding column before saving
        df_target_to_save = df_target.drop("embedding") 
        df_target_to_save.write_csv(output_pred_path)
        logger.info(f"Predictions and sampling results saved, overwriting: {output_pred_path}")
    except Exception as e:
        logger.exception(f"Failed to save prediction results to {output_pred_path}")

    logger.info("Script finished.")


if __name__ == "__main__":
    main()

"""

# Example Usage:
# python recommend_system/script/fit_model.py \
#   --preference-dir data/preference \
#   --background-file data/arxiv_background \
#   --target-file data/arxiv_latest \
#   --neg-ratio 5.0 \
#   -k 5 \
#   --output-model models/preference_lr.joblib \
#   -o data/arxiv_latest_predicted.csv 

""" 