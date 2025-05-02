#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cache Embeddings Script

This script reads the latest arXiv paper data and embeddings, filters papers
based on a score threshold, converts their embeddings to float8, encodes them
in Base64, and appends them to monthly cache files (YYYY-MM.csv) based on
the paper's publication date, ensuring uniqueness within each file.

It also collects papers with scores in [SCORE_THRESHOLD, 0.5) as negative pool
candidates, saving their IDs to monthly CSV files in the neg_pool directory.
"""

import polars as pl
import numpy as np
import base64
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
from loguru import logger
import io

# Attempt to import the float8 type from ml_dtypes
try:
    # We'll use e4m3fn, common for ML. e5m2 is another option.
    from ml_dtypes import float8_e4m3fn as target_float8_dtype
    FLOAT8_TYPE_NAME = "float8_e4m3fn"
except ImportError:
    logger.error("Required library 'ml_dtypes' not found.")
    logger.error("Please install it using: pip install ml_dtypes")
    sys.exit(1)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
LATEST_CSV_PATH = DATA_DIR / "arxiv_latest.csv"
LATEST_NPY_PATH = DATA_DIR / "arxiv_latest.npy"
EMBED_CACHE_DIR = DATA_DIR / "arxiv_embed"
NEG_POOL_DIR = DATA_DIR / "neg_pool"  # 添加负样本池目录
SCORE_THRESHOLD = 0.35
NEG_POOL_UPPER_BOUND = 0.5  # 负样本池的上界

# Define the schema for the cache file to ensure types are correct
CACHE_SCHEMA = {"id": pl.Utf8, "embed": pl.Utf8}
NEG_POOL_SCHEMA = {"id": pl.Utf8}  # 负样本池只需要存储ID

# --- Logger Configuration ---
logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=log_format, level="INFO")
# Uncomment for more detailed debugging output
# logger.add(sys.stderr, format=log_format, level="DEBUG")

# --- Functions ---

def setup_directories():
    """Ensures the cache and neg_pool directories exist."""
    try:
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        NEG_POOL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured cache directory exists: {EMBED_CACHE_DIR}")
        logger.info(f"Ensured negative pool directory exists: {NEG_POOL_DIR}")
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)

def load_latest_data(csv_path: Path, npy_path: Path) -> tuple[pl.DataFrame | None, np.ndarray | None]:
    """Loads the latest paper data (including 'date' column) and embeddings."""
    logger.info(f"Loading latest data from {csv_path} and {npy_path}")
    # Define potential schemas, trying to include 'date' column
    schema_with_date = {"id": pl.Utf8, "score": pl.Float64, "date": pl.Utf8}
    schema_without_date = {"id": pl.Utf8, "score": pl.Float64}

    df = None
    try:
        # Try loading with the 'date' column first
        logger.debug("Attempting to load CSV with 'date' column.")
        try:
            df = pl.read_csv(csv_path, schema_overrides=schema_with_date)
            # Attempt to parse the 'date' string to actual date/datetime
            try:
                 # Parse date with timezone awareness
                 df = df.with_columns(
                     pl.col("date").str.to_datetime(strict=False, format=None, time_zone="UTC").alias("date_dt")
                 )
                 logger.info("Successfully parsed 'date' column to datetime with UTC timezone.")
            except Exception as parse_err:
                 logger.warning(f"Could not auto-parse 'date' column to datetime: {parse_err}. Will use string or fallback logic.")
                 df = df.rename({"date": "date_str"}) # Keep string version if needed

            if "date" not in df.columns and "date_dt" not in df.columns and "date_str" not in df.columns:
                 logger.warning("Column 'date' not found or created. Will use current month for caching.")
                 df = df.with_columns(pl.lit(None).alias("date_dt")) # Add null column if missing

        except pl.ColumnNotFoundError: # If 'date' column doesn't exist
            logger.warning("Column 'date' not found. Loading without it. Will use current month for caching.")
            df = pl.read_csv(csv_path, schema_overrides=schema_without_date)
            df = df.with_columns(pl.lit(None).alias("date_dt")) # Add null column

        logger.info(f"Loaded DataFrame with shape: {df.shape}. Columns: {df.columns}")
        if "id" not in df.columns or "score" not in df.columns:
            logger.error("CSV file missing required columns 'id' or 'score'.")
            return None, None

    except Exception as e:
        logger.error(f"Failed to load or parse CSV file {csv_path}: {e}")
        return None, None

    try:
        embeddings = np.load(npy_path)
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to load NPY file {npy_path}: {e}")
        return None, None

    # Validation: Check if dimensions match
    if df.shape[0] != embeddings.shape[0]:
        logger.error(f"Mismatch between CSV rows ({df.shape[0]}) and NPY embeddings ({embeddings.shape[0]})")
        return None, None

    logger.success("Successfully loaded and validated latest data.")
    return df, embeddings

def convert_and_encode(vector: np.ndarray) -> str | None:
    """Converts a numpy vector to float8, then encodes as Base64."""
    try:
        # Ensure input is a numpy array
        if not isinstance(vector, np.ndarray):
             vector = np.array(vector) # Attempt conversion if needed

        # Convert to target float8 type
        float8_vector = vector.astype(target_float8_dtype)

        # Get the bytes representation
        vector_bytes = float8_vector.tobytes()

        # Encode bytes as Base64
        base64_bytes = base64.b64encode(vector_bytes)

        # Decode Base64 bytes to a UTF-8 string
        base64_string = base64_bytes.decode('utf-8')
        return base64_string
    except Exception as e:
        logger.error(f"Error during embedding conversion/encoding: {e}")
        return None

def process_papers(df: pl.DataFrame, embeddings: np.ndarray) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """
    Filters papers by score, converts/encodes embeddings for high-scoring papers,
    and collects IDs for papers in the [SCORE_THRESHOLD, NEG_POOL_UPPER_BOUND) range as negative examples.
    
    Returns:
        tuple: (embed_results, neg_pool_results)
            - embed_results: Dict mapping month strings to lists of {id, embed} dicts
            - neg_pool_results: Dict mapping month strings to lists of paper IDs
    """
    logger.info(f"Processing papers with score > {SCORE_THRESHOLD} for embeddings")
    logger.info(f"Collecting papers with score in [{SCORE_THRESHOLD}, {NEG_POOL_UPPER_BOUND}) for negative pool")

    # Filter DataFrames
    high_score_df = df.filter(pl.col("score") > SCORE_THRESHOLD)
    neg_pool_df = df.filter(
        (pl.col("score") >= SCORE_THRESHOLD) & 
        (pl.col("score") < NEG_POOL_UPPER_BOUND)
    )
    
    logger.info(f"Found {high_score_df.shape[0]} papers above threshold for embedding")
    logger.info(f"Found {neg_pool_df.shape[0]} papers in negative pool score range")

    if high_score_df.is_empty() and neg_pool_df.is_empty():
        logger.info("No papers met any criteria. Nothing to process.")
        return {}, {}

    # Create a mapping from paper ID to its original index AND parsed date
    id_to_data = {
        row["id"]: {
            "index": i,
            "date_dt": row.get("date_dt") # Use parsed date ('date_dt') if available
        }
        for i, row in enumerate(df.iter_rows(named=True))
    }
    logger.debug(f"Created ID-to-data map for {len(id_to_data)} papers.")

    # ----- Process high score papers for embeddings -----
    embed_results: dict[str, list[dict]] = {}
    neg_pool_results: dict[str, list[str]] = {}
    now_year_month = datetime.now(timezone.utc).strftime("%Y-%m") # Use UTC time for fallback
    
    # Statistics counters
    processed_count = 0
    skipped_count = 0
    fallback_count = 0
    neg_processed_count = 0
    neg_skipped_count = 0

    # Process papers for embedding cache
    for row in high_score_df.iter_rows(named=True):
        paper_id = row["id"]
        if paper_id in id_to_data:
            paper_data = id_to_data[paper_id]
            original_index = paper_data["index"]
            publication_date = paper_data["date_dt"]
            embedding_vector = embeddings[original_index]

            # Determine the year-month string
            year_month = get_year_month(publication_date, paper_id, now_year_month)
            if year_month is None:
                fallback_count += 1
                year_month = now_year_month

            # Convert and encode
            b64_embed = convert_and_encode(embedding_vector)
            if b64_embed:
                entry = {"id": paper_id, "embed": b64_embed}
                if year_month not in embed_results:
                    embed_results[year_month] = []
                embed_results[year_month].append(entry)
                processed_count += 1
            else:
                logger.warning(f"Skipping paper {paper_id} due to conversion/encoding error.")
                skipped_count += 1
        else:
            logger.warning(f"Paper ID {paper_id} from filtered data not found in original index map. Skipping.")
            skipped_count += 1

    # Process papers for negative pool
    for row in neg_pool_df.iter_rows(named=True):
        paper_id = row["id"]
        if paper_id in id_to_data:
            paper_data = id_to_data[paper_id]
            publication_date = paper_data["date_dt"]
            
            # Determine the year-month string for negative pool
            year_month = get_year_month(publication_date, paper_id, now_year_month)
            if year_month is None:
                fallback_count += 1
                year_month = now_year_month

            # Add to negative pool results
            if year_month not in neg_pool_results:
                neg_pool_results[year_month] = []
            neg_pool_results[year_month].append(paper_id)
            neg_processed_count += 1
        else:
            logger.warning(f"Paper ID {paper_id} from negative pool data not found. Skipping.")
            neg_skipped_count += 1

    logger.info(f"Successfully processed {processed_count} papers for embeddings, skipped {skipped_count}.")
    logger.info(f"Collected {neg_processed_count} paper IDs for negative pool, skipped {neg_skipped_count}.")
    logger.info(f"Used fallback month in {fallback_count} cases.")
    
    return embed_results, neg_pool_results

def get_year_month(date_obj, paper_id, fallback_ym=None):
    """Helper to extract year-month string from a date object with error handling."""
    try:
        if date_obj and isinstance(date_obj, (datetime)):
            # Ensure we're working with a timezone-aware datetime or assume UTC
            if date_obj.tzinfo is None:
                logger.warning(f"Date for {paper_id} has no timezone info. Assuming UTC.")
                date_obj = date_obj.replace(tzinfo=timezone.utc)
            return date_obj.strftime("%Y-%m")
        else:
            logger.warning(f"Missing or invalid publication date for {paper_id}. Using fallback month.")
            return fallback_ym
    except ValueError:
        logger.warning(f"Invalid publication date for {paper_id}: {date_obj}. Using fallback month.")
        return fallback_ym

def update_cache_file(cache_dir: Path, year_month: str, new_data_list: list[dict]):
    """Loads existing IDs for a specific month, appends new unique data, and saves."""
    if not new_data_list:
        logger.debug(f"No new data provided for month {year_month}.")
        return

    cache_file = cache_dir / f"{year_month}.csv"
    logger.info(f"Updating cache file: {cache_file} for month {year_month}")

    # Convert list of dicts to DataFrame for easier processing
    new_data_df = pl.DataFrame(new_data_list, schema=CACHE_SCHEMA)

    existing_ids = set()
    file_exists = cache_file.exists()
    file_is_empty = not file_exists or cache_file.stat().st_size == 0

    if file_exists and not file_is_empty:
        try:
            existing_df = pl.read_csv(cache_file, columns=["id"], schema_overrides=CACHE_SCHEMA)
            existing_ids = set(existing_df["id"].to_list())
            logger.info(f"Loaded {len(existing_ids)} existing IDs from cache for {year_month}.")
        except Exception as e:
            logger.warning(f"Failed to load existing IDs from {cache_file}: {e}. Treating as empty/new.")
            file_exists = False
            file_is_empty = True

    # Filter new data to find entries whose IDs are not in the existing cache for this month
    entries_to_add = new_data_df.filter(~pl.col("id").is_in(existing_ids))
    added_count = entries_to_add.shape[0]

    if added_count == 0:
        logger.info(f"No new unique entries to add to the cache for {year_month}.")
        if not file_exists:
             logger.info(f"Writing empty cache file with header for {year_month} as it doesn't exist.")
             pl.DataFrame([], schema=CACHE_SCHEMA).write_csv(cache_file, include_header=True)
        return

    logger.info(f"Found {added_count} new unique entries to append for {year_month}.")

    try:
        write_hdr = not file_exists or file_is_empty
        logger.debug(f"Writing to {cache_file}. File exists: {file_exists}, File empty: {file_is_empty}, Writing header: {write_hdr}")

        if write_hdr:
             entries_to_add.write_csv(cache_file, include_header=True)
             logger.success(f"Successfully created/wrote cache file for {year_month} with {added_count} new entries.")
        else:
             buffer = io.StringIO()
             entries_to_add.write_csv(buffer, include_header=False)
             csv_data_to_append = buffer.getvalue()
             buffer.close()
             with open(cache_file, "a", encoding="utf-8") as f:
                  f.write(csv_data_to_append)
             logger.success(f"Successfully appended {added_count} new entries for {year_month}.")

    except Exception as e:
        logger.error(f"Failed to write/append to cache file {cache_file}: {e}")

def update_neg_pool_file(neg_pool_dir: Path, year_month: str, new_ids: list[str]):
    """Updates the negative pool file for a specific month with new paper IDs."""
    if not new_ids:
        logger.debug(f"No new negative pool IDs for month {year_month}.")
        return
        
    neg_pool_file = neg_pool_dir / f"{year_month}.csv"
    logger.info(f"Updating negative pool file: {neg_pool_file} for month {year_month}")
    
    # Convert list of IDs to DataFrame
    new_ids_df = pl.DataFrame({"id": new_ids}, schema=NEG_POOL_SCHEMA)
    
    existing_ids = set()
    file_exists = neg_pool_file.exists() 
    file_is_empty = not file_exists or neg_pool_file.stat().st_size == 0
    
    if file_exists and not file_is_empty:
        try:
            existing_df = pl.read_csv(neg_pool_file, schema=NEG_POOL_SCHEMA)
            existing_ids = set(existing_df["id"].to_list())
            logger.info(f"Loaded {len(existing_ids)} existing IDs from negative pool for {year_month}.")
        except Exception as e:
            logger.warning(f"Failed to load existing IDs from negative pool {neg_pool_file}: {e}. Treating as empty/new.")
            file_exists = False
            file_is_empty = True
            
    # Filter to find new unique IDs
    ids_to_add = new_ids_df.filter(~pl.col("id").is_in(existing_ids))
    added_count = ids_to_add.shape[0]
    
    if added_count == 0:
        logger.info(f"No new unique IDs to add to negative pool for {year_month}.")
        if not file_exists:
            logger.info(f"Writing empty negative pool file with header for {year_month} as it doesn't exist.")
            pl.DataFrame([], schema=NEG_POOL_SCHEMA).write_csv(neg_pool_file, include_header=True)
        return
        
    logger.info(f"Found {added_count} new unique IDs to append to negative pool for {year_month}.")
    
    try:
        write_hdr = not file_exists or file_is_empty
        
        if write_hdr:
            ids_to_add.write_csv(neg_pool_file, include_header=True)
            logger.success(f"Successfully created/wrote negative pool file for {year_month} with {added_count} new IDs.")
        else:
            buffer = io.StringIO()
            ids_to_add.write_csv(buffer, include_header=False)
            csv_data_to_append = buffer.getvalue()
            buffer.close()
            with open(neg_pool_file, "a", encoding="utf-8") as f:
                f.write(csv_data_to_append)
            logger.success(f"Successfully appended {added_count} new IDs to negative pool for {year_month}.")
            
    except Exception as e:
        logger.error(f"Failed to write/append to negative pool file {neg_pool_file}: {e}")

# --- Main Execution ---
def main():
    """Main function to run the caching process."""
    start_time = datetime.now(timezone.utc)  # Use UTC for consistent timestamps
    logger.info("--- Starting Embeddings Caching Script ---")

    # 1. Ensure directories exist
    setup_directories()

    # 2. Load latest data
    latest_df, latest_embeddings = load_latest_data(LATEST_CSV_PATH, LATEST_NPY_PATH)
    if latest_df is None or latest_embeddings is None:
        logger.error("Failed to load input data. Exiting.")
        sys.exit(1)

    # 3. Process papers: filter by scores, convert/encode embeddings, extract negative pool IDs
    embed_results, neg_pool_results = process_papers(latest_df, latest_embeddings)

    # 4. Update embedding cache files
    if not embed_results:
        logger.info("No papers processed or eligible for embedding cache.")
    else:
        logger.info(f"Updating embedding cache files for {len(embed_results)} months...")
        for year_month, new_entries_list in embed_results.items():
            update_cache_file(EMBED_CACHE_DIR, year_month, new_entries_list)
        logger.info("Finished updating all monthly embedding cache files.")
        
    # 5. Update negative pool files
    if not neg_pool_results:
        logger.info("No papers eligible for negative pool.")
    else:
        logger.info(f"Updating negative pool files for {len(neg_pool_results)} months...")
        for year_month, id_list in neg_pool_results.items():
            update_neg_pool_file(NEG_POOL_DIR, year_month, id_list)
        logger.info("Finished updating all monthly negative pool files.")

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    logger.info(f"--- Embeddings Caching Script Finished in {duration:.2f} seconds ---")

if __name__ == "__main__":
    main() 