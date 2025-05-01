import argparse
import os
import pathlib
import sys
import toml
import numpy as np
import polars as pl
from loguru import logger
import re
import time
from typing import List, Dict, Any, Optional
import arxiv # Import arxiv library
import pytz # For handling timezone in dates
from datetime import date

# --- 设置相对路径导入 ---
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from src.embed import AliCloudEmbed, EmbedBase
except ImportError as e:
    logger.error(f"Failed to import embedding module: {e}")
    sys.exit(1)

# --- 日志配置 ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 配置加载 ---
def load_config() -> dict:
    """Loads the entire config.toml file."""
    config_path = project_root / "config.toml"
    config = {}
    if config_path.is_file():
        logger.info(f"Loading configuration from: {config_path}")
        try:
            config = toml.load(config_path)
            logger.info(f"Configuration loaded successfully.")
        except Exception as e:
            logger.exception(f"Error loading config file {config_path}: {e}")
            config = {} # Return empty dict on error
    else:
        logger.warning(f"Configuration file not found at {config_path}. Required configs might be missing.")
    return config

# --- Helper Function for ID Normalization ---
def normalize_arxiv_id(raw_id: str) -> str:
    """Normalizes arXiv ID by removing version and padding the suffix with zeros."""
    if not isinstance(raw_id, str):
        # Attempt conversion if not string, log warning
        logger.warning(f"Attempting to normalize non-string ID: {raw_id} of type {type(raw_id)}. Converting to string.")
        raw_id = str(raw_id)
    base_id = re.sub(r'v\d+$', '', raw_id.strip())
    parts = base_id.split('.')
    if len(parts) == 2:
        year_month = parts[0]
        suffix = parts[1]
        # Basic check for common format YYMM or YYYY.MM
        if not (len(year_month) == 4 or (len(year_month) == 7 and year_month[4] == '.')):
             logger.warning(f"Unexpected format before dot in ID {raw_id}. Returning base ID {base_id}.")
             return base_id
        try:
            # Normalize the part after the dot to 5 digits with leading zeros
            normalized_suffix = f"{int(suffix):05d}"
            normalized_id = f"{year_month}.{normalized_suffix}"
            # logger.debug(f"Normalized {raw_id} -> {normalized_id}")
            return normalized_id
        except ValueError:
            logger.warning(f"Could not normalize numeric suffix for ID {raw_id}. Returning base ID {base_id}.")
            return base_id
    # logger.debug(f"ID {raw_id} did not match expected format YYMM.##### or YYYY.MM.##### for normalization. Returning base ID {base_id}.")
    return base_id

# --- ArXiv Fetching Function ---
def fetch_details_by_ids(id_list: List[str], delay_seconds: float, individual_retry_delay: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """Fetches paper details for a list of arXiv IDs, attempting bulk first, then individual retries for missing ones."""
    if not id_list:
        return {}

    logger.info(f"Attempting to fetch details for {len(id_list)} unique arXiv IDs (bulk query first)...")
    # Clean and Normalize IDs for bulk query
    # Create a map from normalized ID back to the first original ID encountered for it
    original_id_map = {normalize_arxiv_id(paper_id): paper_id for paper_id in reversed(id_list)} # Use reversed so first occurrence wins if multiple normalize to same
    unique_normalized_ids_for_query = sorted(list(original_id_map.keys()))
    logger.debug(f"Querying with {len(unique_normalized_ids_for_query)} unique normalized IDs...")

    client = arxiv.Client(page_size=100, delay_seconds=delay_seconds)
    # Query using the normalized IDs
    search = arxiv.Search(id_list=unique_normalized_ids_for_query, max_results=len(unique_normalized_ids_for_query))

    results_map: Dict[str, Dict[str, Any]] = {} # Keyed by NORMALIZED ID
    fetched_count_bulk = 0
    try:
        results_iterator = client.results(search)
        for result in results_iterator:
            fetched_count_bulk += 1
            arxiv_id_raw = result.entry_id.split('/')[-1]
            # Normalize the ID fetched from the result
            normalized_fetched_id = normalize_arxiv_id(arxiv_id_raw)

            # Check if the normalized fetched ID was one we requested
            if normalized_fetched_id not in unique_normalized_ids_for_query:
                logger.warning(f"Received unexpected normalized ID {normalized_fetched_id} (from {arxiv_id_raw}) in bulk results. Skipping.")
                continue

            authors = '; '.join([author.name for author in result.authors])
            published_date = result.published
            if published_date and published_date.tzinfo is None:
                 published_date = pytz.utc.localize(published_date)
            published_date_iso = published_date.isoformat() if published_date else None

            details = {
                'type': 'arxiv',
                # Store the normalized ID used for mapping
                'id': normalized_fetched_id,
                'title': result.title.replace('\n', ' ').strip(),
                'authors': authors,
                'abstract': result.summary.replace('\n', ' ').strip() if result.summary else "",
                'date': published_date_iso,
                'primary_category': result.primary_category,
                'pdf_url': result.pdf_url
            }
            # Use the normalized ID as the key in the results map
            if normalized_fetched_id in results_map:
                logger.warning(f"Duplicate normalized ID {normalized_fetched_id} encountered in bulk results. Overwriting previous entry.")
            results_map[normalized_fetched_id] = details
            if fetched_count_bulk % 50 == 0:
                 logger.info(f"[Bulk] Processed {fetched_count_bulk}/{len(unique_normalized_ids_for_query)} potential results...")

    except ConnectionError as ce:
        logger.error(f"[Bulk] Connection error during fetching: {ce}. Proceeding to individual retries if needed.")
    except Exception as e:
        logger.exception("[Bulk] An unexpected error occurred during bulk fetching by ID. Proceeding to individual retries if needed.")

    logger.info(f"[Bulk] Finished bulk fetch. Found details for {len(results_map)} normalized IDs initially.")

    # --- Fallback: Individual retries for missing IDs ---
    # Identify missing IDs based on the normalized IDs we queried with
    missing_normalized_ids = set(unique_normalized_ids_for_query) - set(results_map.keys())

    if missing_normalized_ids:
        logger.warning(f"{len(missing_normalized_ids)} normalized IDs were not found in the bulk query. Attempting individual fetches...")
        fetched_count_retry = 0
        retry_client = arxiv.Client(page_size=1, delay_seconds=individual_retry_delay)

        # Iterate through the missing *normalized* IDs
        for missing_norm_id in sorted(list(missing_normalized_ids)):
            # We still query ArXiv using the normalized ID form
            logger.debug(f"[Retry] Attempting fetch for normalized ID: {missing_norm_id}")
            try:
                individual_search = arxiv.Search(id_list=[missing_norm_id], max_results=1)
                retry_results = list(retry_client.results(individual_search))

                if retry_results:
                    result = retry_results[0]
                    arxiv_id_raw = result.entry_id.split('/')[-1]
                    # Normalize the ID from the retry result
                    normalized_fetched_id_retry = normalize_arxiv_id(arxiv_id_raw)

                    # Compare normalized IDs
                    if normalized_fetched_id_retry == missing_norm_id:
                        authors = '; '.join([author.name for author in result.authors])
                        published_date = result.published
                        if published_date and published_date.tzinfo is None:
                             published_date = pytz.utc.localize(published_date)
                        published_date_iso = published_date.isoformat() if published_date else None

                        details = {
                            'type': 'arxiv',
                            # Store the matched normalized ID
                            'id': normalized_fetched_id_retry,
                            'title': result.title.replace('\n', ' ').strip(),
                            'authors': authors,
                            'abstract': result.summary.replace('\n', ' ').strip() if result.summary else "",
                            'date': published_date_iso,
                            'primary_category': result.primary_category,
                            'pdf_url': result.pdf_url
                        }
                        # Add to results map using the normalized ID as key
                        if normalized_fetched_id_retry in results_map:
                             logger.warning(f"Duplicate normalized ID {normalized_fetched_id_retry} encountered during retry. Overwriting previous entry.")
                        results_map[normalized_fetched_id_retry] = details
                        fetched_count_retry += 1
                        logger.info(f"[Retry] Successfully fetched details for {missing_norm_id}")
                    else:
                        # This case should be less likely now after normalization
                        logger.warning(f"[Retry] Fetched result for query {missing_norm_id}, but normalized fetched ID {normalized_fetched_id_retry} doesn't match normalized query ID.")
                else:
                     logger.warning(f"[Retry] No results found for individual ID query: {missing_norm_id}")

            except Exception as e:
                 logger.exception(f"[Retry] Error fetching individual ID {missing_norm_id}")
                 # Continue to next retry

        logger.info(f"[Retry] Finished individual retries. Found details for an additional {fetched_count_retry} IDs.")

    # Final summary log
    final_fetched_count = len(results_map)
    total_requested = len(unique_normalized_ids_for_query) # Compare against normalized IDs
    final_missing_count = total_requested - final_fetched_count
    logger.info(f"Fetch process complete. Successfully obtained details for {final_fetched_count}/{total_requested} unique normalized IDs.")
    if final_missing_count > 0:
        final_missing_normalized_ids = set(unique_normalized_ids_for_query) - set(results_map.keys())
        # Map back to original IDs for reporting
        missing_original_ids = [original_id_map[norm_id] for norm_id in final_missing_normalized_ids if norm_id in original_id_map]
        logger.warning(f"Still could not fetch details for {final_missing_count} IDs (examples: {missing_original_ids[:10]}...). Check if these IDs are valid/public.")

    # The map now uses normalized IDs as keys. The join needs to happen on normalized IDs.
    return results_map

# --- Main Execution ---
def main():
    # Load the entire config
    full_config = load_config()
    # Get specific sections, defaulting to empty dict if section is missing
    embed_cfg = full_config.get('embedding', {})
    bootstrap_cfg = full_config.get('bootstrap', {})

    parser = argparse.ArgumentParser(
        description="Build bootstrap dataset: Fetch arXiv details for IDs in a CSV, combine with preference, and generate embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-bootstrap-csv", "-i", type=str,
                        default=bootstrap_cfg.get('bootstrap_input_csv'),
                        help="Path to the input CSV file with columns 'id' and 'preference' ('like'/'dislike').")
    parser.add_argument("--output-csv", "-o", type=str,
                        default=bootstrap_cfg.get('output_csv'),
                        help="Path to save the enriched CSV output. Overrides config and default naming.")
    parser.add_argument("--output-npy", "-on", type=str,
                        default=bootstrap_cfg.get('output_npy'),
                        help="Path to save the output NumPy embeddings (.npy). Overrides config and default naming.")
    parser.add_argument("--arxiv-delay", type=float,
                        default=bootstrap_cfg.get('arxiv_fetch_delay', 5.0),
                        help="Delay in seconds between ArXiv API requests when fetching details.")

    args = parser.parse_args()

    # --- 参数校验和路径处理 (使用简化变量名) ---
    bootstrap_csv_in = args.input_bootstrap_csv
    enriched_csv_out = args.output_csv
    embed_npy_out = args.output_npy
    arxiv_delay = args.arxiv_delay

    if not bootstrap_csv_in:
        logger.error("Input bootstrap CSV path is required. Provide via -i/--input-bootstrap-csv or config.toml [bootstrap].bootstrap_input_csv")
        return

    input_csv_path = pathlib.Path(bootstrap_csv_in).resolve()
    if not input_csv_path.is_file():
        logger.error(f"Input bootstrap CSV file not found: {input_csv_path}")
        return

    # Determine output paths: CLI > Config > Default derivation
    if enriched_csv_out: # CLI has highest priority
        output_csv_path = pathlib.Path(enriched_csv_out).resolve()
    elif bootstrap_cfg.get('output_csv'): # Then config file
        output_csv_path = pathlib.Path(bootstrap_cfg['output_csv']).resolve()
    else: # Finally, derive from input
        output_csv_path = input_csv_path.with_name(f"{input_csv_path.stem}_enriched.csv")
        logger.info(f"Output CSV path not specified, derived: {output_csv_path}")

    if embed_npy_out: # CLI has highest priority
        output_npy_path = pathlib.Path(embed_npy_out).resolve()
    elif bootstrap_cfg.get('output_npy'): # Then config file
        output_npy_path = pathlib.Path(bootstrap_cfg['output_npy']).resolve()
    else: # Finally, derive from input
        output_npy_path = input_csv_path.with_name(f"{input_csv_path.stem}_embeddings.npy")
        logger.info(f"Output NPY path not specified, derived: {output_npy_path}")
        
    # Ensure .npy suffix for npy path if manually specified
    if output_npy_path.suffix != '.npy':
         logger.warning(f"Output NPY path {output_npy_path} does not end with .npy. Appending suffix.")
         output_npy_path = output_npy_path.with_suffix('.npy')

    logger.info(f"Input Bootstrap CSV: {input_csv_path}")
    logger.info(f"Using Output Enriched CSV: {output_csv_path}")
    logger.info(f"Output Embeddings NPY: {output_npy_path}")

    # --- 读取和验证输入 CSV ---
    try:
        logger.info(f"Reading bootstrap CSV file: {input_csv_path}")
        # Explicitly set dtype for 'id' column to Utf8 (string)
        # Use schema_overrides instead of deprecated dtypes
        df_bootstrap = pl.read_csv(input_csv_path, schema_overrides={'id': pl.Utf8})

        # Use hardcoded column names 'id' and 'preference'
        required_cols = {'id', 'preference'}
        if not required_cols.issubset(df_bootstrap.columns):
            logger.error(f"Input CSV missing required columns. Need: {required_cols}, Found: {df_bootstrap.columns}")
            return

        # Validate preference values using hardcoded name
        valid_prefs = {'like', 'dislike'}
        invalid_prefs = df_bootstrap.filter(pl.col('preference').is_in(valid_prefs).not_()) # Find rows NOT in valid_prefs
        if not invalid_prefs.is_empty():
             logger.error(f"Invalid values found in preference column 'preference'. Only 'like' or 'dislike' allowed.")
             logger.error(f"Invalid rows:\n{invalid_prefs}")
             return
             
        # Get unique, non-null ArXiv IDs using hardcoded name
        unique_ids = df_bootstrap['id'].drop_nulls().unique().to_list()
        if not unique_ids:
             logger.error(f"No valid ArXiv IDs found in column 'id'.")
             return
             
        logger.info(f"Found {len(unique_ids)} unique ArXiv IDs in bootstrap file.")

    except Exception as e:
        logger.exception(f"Failed to read or validate bootstrap CSV file: {input_csv_path}")
        return

    # --- 获取 ArXiv 详细信息 ---
    details_map = fetch_details_by_ids(unique_ids, arxiv_delay)

    if not details_map:
        logger.error("Failed to fetch any details from ArXiv. Cannot proceed.")
        return

    # --- 合并数据 --- 
    enriched_data = []
    missing_details_count = 0
    try:
        logger.info("Combining bootstrap preferences with fetched ArXiv details...")
        # Convert map values to DataFrame for efficient join
        # Ensure the 'id' column in df_details contains the normalized IDs used as keys in details_map
        df_details = pl.DataFrame(list(details_map.values())) # df_details['id'] is normalized

        # Normalize the 'id' column in bootstrap df for joining
        # Also store the original ID for final output
        df_bootstrap = df_bootstrap.with_columns([
            pl.col('id').alias("original_id"), # Keep original ID
            # Use map_elements instead of apply for element-wise function application
            pl.col('id').map_elements(normalize_arxiv_id, return_dtype=pl.Utf8).alias("normalized_join_id") 
        ])

        # Perform the join using normalized IDs
        df_enriched = df_bootstrap.join(
            df_details,
            left_on="normalized_join_id",
            right_on="id", # 'id' in df_details is the normalized one
            how="left",
            suffix="_details" # Add suffix to distinguish columns from df_details if needed (like 'id_details')
        )

        # Select and reorder columns, keep original preference, drop temporary join key
        # Use the original 'id' from df_bootstrap for the final output id column
        final_cols_ordered = [
            'type',
            'original_id', # Use the original ID column
            'title',
            'authors',
            'date',
            'primary_category',
            'pdf_url',
            'preference',
            'abstract'
            ]

        # Select only columns that actually exist after the join, handling potential nulls from left join
        # Rename 'original_id' back to 'id' for the final output
        select_expressions = []
        rename_map = {"original_id": "id"} # Rename map for final output

        for col_name in final_cols_ordered:
             # Handle the rename case
             source_col_name = "original_id" if col_name == "id" else col_name
             # Check if the source column exists in the joined dataframe
             if source_col_name in df_enriched.columns:
                  select_expressions.append(pl.col(source_col_name).alias(rename_map.get(source_col_name, source_col_name)))
             elif col_name == 'type': # Add type column manually if not from details
                 select_expressions.append(pl.lit('arxiv').alias('type'))
             # else: # Column doesn't exist, skip it (or handle as needed)
             #    logger.warning(f"Column '{col_name}' not found in enriched data, skipping.")


        # df_enriched_final = df_enriched.select(available_select_exprs)
        df_enriched_final = df_enriched.select(select_expressions)


        # Log missing details based on null title after join
        missing_details_count = df_enriched.filter(pl.col("title").is_null()).height
        if missing_details_count > 0:
            logger.warning(f"{missing_details_count} entries from bootstrap CSV did not have corresponding details fetched from ArXiv (based on normalized ID matching). These rows will have null details.")
            # Log some IDs that failed the join
            # missing_joined_ids = df_enriched.filter(pl.col("title").is_null())['original_id'].to_list()
            # logger.warning(f"Original IDs without details after join: {missing_joined_ids[:10]}...")


    except Exception as e:
        logger.exception("Failed to combine bootstrap data with ArXiv details.")
        return
        
    # Filter out rows where details couldn't be fetched if necessary for embedding
    df_valid = df_enriched_final.filter(pl.col("title").is_not_null()) 
    if df_valid.is_empty():
         logger.error("No valid entries remaining after merging with ArXiv details. Cannot generate embeddings.")
         return

    # --- 保存丰富后的 CSV --- 
    try:
        logger.info(f"Saving enriched data to {output_csv_path}...")
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_valid.write_csv(output_csv_path)
        logger.info(f"Enriched CSV saved successfully ({df_valid.height} rows).")
    except Exception as e:
        logger.exception(f"Failed to save enriched CSV to {output_csv_path}")
        # Decide whether to stop or continue to embedding
        return 

    # --- 生成 Embeddings --- 
    # 只使用同时有标题和摘要的行
    df_for_embed = df_valid.filter(
        pl.col('title').is_not_null() & 
        pl.col('abstract').is_not_null()
    )
    
    if df_for_embed.height == 0:
        logger.warning("No rows with both title and abstract found. Cannot generate embeddings.")
        return
    
    # 使用select获取列，然后用rows()方法获取行数据
    rows_data = df_for_embed.select([
        pl.col('title'),
        pl.col('abstract')
    ]).rows()
    
    # 使用列表推导式创建嵌入文本
    embedding_texts = [f"Title: {row[0]}\nAbstract: {row[1]}" for row in rows_data]
    
    logger.info(f"Prepared {len(embedding_texts)} documents with 'Title: {{}}\nAbstract: {{}}' format for embedding.")

    # Initialize Embedding Model
    # Get API key from config, check if it's set to "env"
    api_key_config = embed_cfg.get('api_key') # Get from embed_cfg
    api_key = None
    if api_key_config == "env":
        logger.info("API key configured as 'env', attempting to read from EMBEDDING_API_KEY environment variable.")
        api_key = os.getenv('EMBEDDING_API_KEY')
        if not api_key:
            logger.error("API key set to 'env' in config, but EMBEDDING_API_KEY environment variable is not set or empty.")
            return
    elif api_key_config:
        api_key = api_key_config # Use the value from config directly
    else:
        # api_key_config is None or empty
        logger.error("AliCloud API key missing. Provide it in config.toml [embedding].api_key or set it to 'env' and export EMBEDDING_API_KEY.")
        return

    # api_key should now hold the actual key or the script would have exited
    base_url = embed_cfg.get('base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = embed_cfg.get('model', "text-embedding-v3")
    dim = embed_cfg.get('dim', 1024)

    try:
        embedder: EmbedBase = AliCloudEmbed(
            api_key=api_key, base_url=base_url, model=model, dim=dim
        )
    except Exception as e:
        logger.exception("Failed to initialize embedding model.")
        return

    # Generate
    try:
        logger.info("Generating embeddings using 'Title: {}\nAbstract: {}' format...")
        embeddings = embedder.embed(embedding_texts)
        logger.info(f"Successfully generated embeddings. Shape: {embeddings.shape}")
    except Exception as e:
        logger.exception("Failed to generate embeddings.")
        return

    if embeddings.shape[0] != len(embedding_texts):
         logger.warning(f"Number of embeddings ({embeddings.shape[0]}) != number of input texts ({len(embedding_texts)}). Output might be incomplete.")

    # --- 保存 Embeddings NPY --- 
    try:
        logger.info(f"Saving embeddings to {output_npy_path}...")
        output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为fp16精度再保存
        embeddings_fp16 = embeddings.astype(np.float16)
        original_size = embeddings.nbytes / (1024 * 1024)
        fp16_size = embeddings_fp16.nbytes / (1024 * 1024)
        logger.info(f"Converting embeddings to fp16 precision. Original size: {original_size:.2f}MB, fp16 size: {fp16_size:.2f}MB")
        
        # 保存为fp16精度的npy文件
        np.save(str(output_npy_path), embeddings_fp16, allow_pickle=False)
        logger.info("Embeddings NPY file saved successfully in fp16 precision.")
    except Exception as e:
        logger.exception(f"Failed to save embeddings NPY to {output_npy_path}")
        
    # 确保写入CSV与嵌入顺序一致
    if df_for_embed.height != df_valid.height:
        # 如果过滤后行数不一致，需要更新CSV
        logger.warning(f"Number of valid embedding documents ({df_for_embed.height}) != number of enriched rows ({df_valid.height}).")
        logger.warning(f"Overwriting enriched CSV with only rows used for embedding to maintain consistency.")
        df_for_embed.write_csv(output_csv_path)
        logger.info(f"Updated enriched CSV saved with {df_for_embed.height} rows with valid title and abstract.")

if __name__ == "__main__":
    main() 