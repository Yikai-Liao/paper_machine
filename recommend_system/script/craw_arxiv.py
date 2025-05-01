import argparse
# Removed standard logging, using loguru instead
# import logging 
import os
import pathlib
import sys # Import sys for stderr configuration
import toml # Import toml for config file parsing
import re # Import re for cleaning arxiv id
import time # Import time for potential manual delays if needed
from datetime import datetime, timedelta, time, date # Import date type hint
from typing import List, Optional, Dict, Any # Keep typing for clarity
# Use dateutil for flexible time string parsing
from dateutil import parser as date_parser
from dateutil import tz as date_tz

import arxiv
import polars as pl
import pytz # Keep pytz for explicit UTC handling where needed

# Import and configure Loguru
from loguru import logger
logger.remove() # Remove default handler to avoid duplicate logs if run multiple times
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- Configuration Loading ---
# Removed DEFAULT_CONFIG dictionary
# DEFAULT_CONFIG = {
#     'days': 1,
#     'category': 'cs.LG', 
#     'cutoff_time': '00:00 UTC', 
#     'batch_size': 1000,
#     'output_file': './arxiv_papers_fallback.csv' 
# }

def load_config() -> dict:
    """Loads configuration from config.toml located in the script's parent directory.
       Looks for [arxiv_crawler] section.
       Returns an empty dict if config is not found or invalid.
    """
    script_dir = pathlib.Path(__file__).parent
    config_path = script_dir.parent / "config.toml"
    config = {} 

    if config_path.is_file():
        logger.info(f"Loading configuration from: {config_path}")
        try:
            loaded_config = toml.load(config_path)
            config = loaded_config.get('arxiv_crawler', {})
            # Handle single or list of categories from config
            if 'category' in config and isinstance(config['category'], str):
                config['category'] = [config['category']]
            logger.info(f"Loaded config from [arxiv_crawler] section: {config}")
        except toml.TomlDecodeError as e:
            logger.exception(f"Error decoding config file {config_path}: {e}")
            logger.warning("Configuration file invalid. Proceeding without config defaults.")
            config = {} 
        except Exception as e:
            logger.exception(f"Error loading config file {config_path}: {e}")
            logger.warning("Failed to load configuration. Proceeding without config defaults.")
            config = {} 
    else:
        logger.warning(f"Configuration file not found at {config_path}. Proceeding without config defaults.")
        
    return config

# --- Removed Time Parsing and Cutoff Calculation ---
# def parse_cutoff_time_string(time_str: str) -> time:
#     ...
# def get_cutoff_datetime(cutoff_time_utc: time) -> datetime:
#     ...

# --- ArXiv Crawling ---
def crawl_arxiv(category: str, start_query_date: date, end_query_date: date, max_results_to_return: int, delay_seconds: float) -> list[dict]:
    """
    Crawls arXiv for papers submitted within a specific category and date range.
    Uses the submittedDate query for server-side filtering (full days).
    Uses arxiv.Client for automatic delay handling. Extracts richer metadata.

    Args:
        category: ArXiv category (e.g., 'cs.AI').
        start_query_date: The start date for the query range (inclusive).
        end_query_date: The end date for the query range (inclusive).
        max_results_to_return: Max number of results to return from the arXiv API query.
        delay_seconds: Delay between API requests handled by the client.

    Returns:
        A list of dictionaries, each containing paper metadata.
    """
    # Instantiate client with delay
    client = arxiv.Client(page_size=100, delay_seconds=delay_seconds)

    # Format dates for Arxiv submittedDate query (YYYYMMDD000000 to YYYYMMDD235959)
    start_str = start_query_date.strftime('%Y%m%d') + '000000'
    end_str = end_query_date.strftime('%Y%m%d') + '235959'
    
    # Construct the query using submittedDate for the full day range
    query = f'cat:{category} AND submittedDate:[{start_str} TO {end_str}]'
    logger.info(f"Constructed arXiv query for full days: {query}")

    # max_results argument limits the API query results
    max_results_query = float('inf') if max_results_to_return <= 0 else max_results_to_return
    logger.info(f"Setting max_results for arXiv query: {max_results_query}")
    logger.info(f"Setting client delay between requests: {delay_seconds}s")

    search = arxiv.Search(
        query=query,
        max_results=max_results_query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results_iterator = client.results(search)

    papers_data = []
    # Updated log message
    logger.info(f"Fetching papers for category '{category}' submitted between {start_query_date.isoformat()} and {end_query_date.isoformat()} (limit: {max_results_query})...")

    count = 0
    fetched_count = 0
    try:
        for result in results_iterator:
            fetched_count += 1
            
            # Metadata extraction remains the same
            authors = '; '.join([author.name for author in result.authors])
            arxiv_id_raw = result.entry_id.split('/')[-1]
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id_raw)
            
            published_date = result.published
            if published_date and published_date.tzinfo is None:
                 published_date = pytz.utc.localize(published_date)
            published_date_iso = published_date.isoformat() if published_date else None

            papers_data.append({
                'type': 'arxiv',
                'id': arxiv_id,
                'title': result.title.replace('\n', ' ').strip(),
                'authors': authors,
                'abstract': result.summary.replace('\n', ' ').strip() if result.summary else "",
                'date': published_date_iso,
                'primary_category': result.primary_category,
                'pdf_url': result.pdf_url
            })
            count += 1
            if count % 100 == 0:
                 logger.info(f"Collected {count} papers (processed {fetched_count} total)...")

    except ConnectionError as ce:
        logger.error(f"Connection error during fetching: {ce}. Check network or arXiv status.")
    except Exception as e:
        logger.exception("An unexpected error occurred during fetching")

    logger.info(f"Finished fetching. Processed {fetched_count} total results from API query. Collected {len(papers_data)} papers.")
    return papers_data


# --- Main Execution ---
def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        # Updated description
        description="Crawl arXiv papers for the last N full UTC days. Reads defaults from ../config.toml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    
    # Arguments use defaults from config
    parser.add_argument("-n", "--days", type=int, 
                        default=config.get('days', 1), # Default to 1 day if not specified
                        help="Number of past full UTC days to crawl (e.g., 1 means yesterday).")
    parser.add_argument("-c", "--category", type=str, 
                        default=config.get('category'), # Expecting a list now, None if not set
                        nargs='+', # Accept one or more categories
                        help="ArXiv category(s) (e.g., 'cs.AI' or 'cs.AI cs.LG').")
    # Removed cutoff-time argument
    # parser.add_argument("--cutoff-time", type=str, ...)
    parser.add_argument("--max-results", "-m", type=int, 
                        default=config.get('max_results', 1000), 
                        help="Maximum number of results to return from the arXiv API query (0 or negative for unlimited).")
    parser.add_argument("--delay", type=float,
                        default=config.get('delay_seconds', 3.0), 
                        help="Delay in seconds between API requests.")
    parser.add_argument("-o", "--output-file", type=str, 
                        default=config.get('output_file'), 
                        help="Output CSV file path.")
    
    args = parser.parse_args()

    # --- Use parsed arguments ---
    days_to_crawl = args.days
    categories = args.category # This is now a list or None
    # Removed cutoff_time_str 
    max_results = args.max_results
    delay_seconds = args.delay 
    output_file = args.output_file

    # Check if essential arguments are provided
    missing_params = []
    if days_to_crawl is None or days_to_crawl <= 0: 
        missing_params.append('days (-n) must be a positive integer')
    if not categories: # Check if list is None or empty
        missing_params.append('category (-c)')
    # Removed cutoff_time_str check
    if output_file is None: missing_params.append('output-file (-o)')
    # max_results and delay have defaults.
    
    if missing_params:
         logger.error(f"Missing or invalid parameters: {', '.join(missing_params)}. Provide them via command line or config.toml [arxiv_crawler] section.")
         parser.print_help()
         return
         
    # --- Calculate Date Range for Full Days ---
    try:
        today_utc_date = datetime.now(pytz.utc).date()
        # End date is yesterday (the last full day)
        end_query_date = today_utc_date - timedelta(days=1)
        # Start date goes back N-1 days from the end date
        start_query_date = end_query_date - timedelta(days=days_to_crawl - 1)
        logger.info(f"Targeting submissions from {start_query_date.isoformat()} to {end_query_date.isoformat()} (inclusive UTC days).")
    except Exception as e:
        logger.exception("Error calculating date range.")
        return

    # Updated log message
    logger.info(f"Running crawl: days={days_to_crawl}, categories={categories}, max_results={max_results}, delay={delay_seconds}s, output='{output_file}'")

    # --- Crawl for each category --- 
    all_papers_data = []
    for category in categories:
        logger.info(f"--- Starting crawl for category: {category} ---")
        try:
            # Call crawl_arxiv for the current category
            papers_data = crawl_arxiv(category, start_query_date, end_query_date, max_results, delay_seconds)
            all_papers_data.extend(papers_data)
            logger.info(f"--- Finished crawl for category: {category}, found {len(papers_data)} papers --- ")
        except Exception as e:
             logger.exception(f"Failed to crawl category {category}")
             # Optionally continue to the next category or break
             # continue 
    
    # Check if any papers were found across all categories
    if not all_papers_data:
        logger.warning("No papers found for the specified criteria across all categories. Exiting.")
        return

    logger.info(f"Total papers collected across all categories: {len(all_papers_data)}")

    # Create Polars DataFrame and save
    try:
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use the combined list of papers
        df = pl.DataFrame(all_papers_data, schema={
             "type": pl.Utf8, 
             "id": pl.Utf8, 
             "title": pl.Utf8, 
             "authors": pl.Utf8, 
             "date": pl.Utf8, 
             "primary_category": pl.Utf8, 
             "pdf_url": pl.Utf8,
             "abstract": pl.Utf8, 
        })
        logger.info(f"Created DataFrame with {len(df)} rows.")
        
        df.write_csv(output_path)
        logger.info(f"Successfully saved data to {output_path}")
        
    except Exception as e:
        logger.exception("Failed to create or save DataFrame")


if __name__ == "__main__":
    main()
