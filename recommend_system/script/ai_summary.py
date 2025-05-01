#!/usr/bin/env python3
"""
AI Summary Script for arXiv Papers

This script reads a CSV file containing arXiv paper information,
filters papers based on type and show status, downloads PDFs from URLs,
extracts text, generates AI summaries using an LLM, and saves the results as Markdown files.
"""

import os
import sys
import asyncio
import argparse
import tempfile
import toml
import polars as pl
from pathlib import Path
import aiohttp
import aiofiles
from loguru import logger
import datetime
import shutil
import multiprocessing
from functools import partial
import arxiv
import time
import random
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
import re

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))
from src.llm_client import LLMClient, LLMClientError
from src.pdf_extractor import extract_text_from_pdf
from src.summary_parser import extract_summary_json, SummaryParseError
from src.template_renderer import TemplateRenderer, TemplateError
from src.utils import save_file

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "post"

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

async def init_llm_client(config):
    """Initialize LLM client from config."""
    try:
        # LLMClient 需要 'openai' 配置节点，但我们的配置在 'summary' 下
        # 我们需要转换配置格式
        summary_config = config.get('summary', {})

        # Get API key, checking for "env" sentinel value
        api_key_from_config = summary_config.get('api_key')
        api_key = None
        if api_key_from_config == "env":
            logger.info("Config specifies 'env' for API key, reading from SUMMARY_API_KEY environment variable.")
            api_key = os.environ.get('SUMMARY_API_KEY')
            if not api_key:
                logger.error("API key set to 'env' in config, but SUMMARY_API_KEY environment variable not found.")
                raise LLMClientError("SUMMARY_API_KEY environment variable is not set.")
            else:
                 logger.info("Successfully loaded API key from SUMMARY_API_KEY environment variable.")
        else:
            logger.info("Using API key directly from config file.")
            api_key = api_key_from_config

        # 创建一个新的配置字典，将 summary 配置转换为 openai 格式
        llm_config = {
            'openai': {
                'base_url': summary_config.get('base_url'),
                'model': summary_config.get('model'),
                'api_key': api_key, # Use the determined API key
                'temperature': summary_config.get('temperature'),
                'top_p': summary_config.get('top_p'),
                'max_tokens': summary_config.get('max_tokens'),
                'reasoning_level': summary_config.get('reasoning_level', None)
            }
        }

        logger.debug(f"Initializing LLM client with model: {llm_config['openai']['model']}")

        # 使用转换后的配置初始化 LLM 客户端
        llm_client = LLMClient(llm_config)
        return llm_client
    except LLMClientError as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        raise

def load_config(config_path: Path):
    """Load configuration from TOML file."""
    try:
        logger.info(f"Loading configuration from {config_path}")
        config = toml.load(str(config_path))
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def load_prompt_instructions(prompt_path: Path) -> str:
    """Load prompt instructions from file."""
    try:
        logger.info(f"Loading prompt instructions from {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load prompt instructions from {prompt_path}: {e}")
        raise

def construct_full_prompt(base_instructions: str, extracted_text: str) -> str:
    """Construct full prompt by combining base instructions and extracted text."""
    # Truncate extracted text if it's too long (LLMs typically have token limits)
    # A more sophisticated approach would be to summarize or chunk the text
    max_chars = 100000  # Arbitrary limit, adjust based on your LLM's context window
    if len(extracted_text) > max_chars:
        logger.warning(f"Extracted text is too long ({len(extracted_text)} chars), truncating to {max_chars} chars")
        extracted_text = extracted_text[:max_chars] + "...[TRUNCATED]"
    
    full_prompt = f"{base_instructions}\n\n{extracted_text}"
    return full_prompt

async def download_pdf(session, paper_id: str, pdf_url: str, output_dir: Path) -> tuple[Path | None, dict | None]:
    """Download a PDF using arxiv package and return its path and metadata."""
    output_path = output_dir / f"{paper_id}.pdf"
    
    # 创建目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Fetching metadata and downloading PDF for {paper_id} using arxiv API")
        
        # 从URL中提取arxiv ID
        if "arxiv.org" in pdf_url:
            # 处理不同格式的arxiv URL
            if "/pdf/" in pdf_url:
                arxiv_id = pdf_url.split("/pdf/")[1].split("v")[0]
            elif "/abs/" in pdf_url:
                arxiv_id = pdf_url.split("/abs/")[1].split("v")[0]
            else:
                arxiv_id = pdf_url.split("/")[-1].split("v")[0]
                
            logger.info(f"Extracted arXiv ID: {arxiv_id} for paper {paper_id}")
        else:
            raise ProcessingError(f"URL does not appear to be an arXiv URL: {pdf_url}")
        
        # 添加随机延迟，避免被反爬虫机制拦截
        delay = random.uniform(1.0, 3.0)
        logger.info(f"Adding delay of {delay:.2f} seconds before API call")
        await asyncio.sleep(delay)
        
        # 使用arxiv包的新API查询论文
        client = arxiv.Client(page_size=1, delay_seconds=1, num_retries=3)
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        
        loop = asyncio.get_event_loop()
        paper_results = await loop.run_in_executor(None, lambda: list(client.results(search)))
        
        if not paper_results:
            raise ProcessingError(f"No paper found with ID {arxiv_id}")
        paper = paper_results[0]
        
        # 提取元数据
        metadata = {
            "title": paper.title.replace('\n', ' ').strip(),
            "published_date": paper.published # This is a datetime object
        }
        logger.info(f"Fetched metadata for {paper_id}: Title - {metadata['title']}, Published - {metadata['published_date']}")
        
        # 添加另一个随机延迟
        delay = random.uniform(1.0, 3.0)
        logger.info(f"Adding delay of {delay:.2f} seconds before download")
        await asyncio.sleep(delay)
        
        # 下载PDF
        await loop.run_in_executor(None, lambda: paper.download_pdf(dirpath=str(output_dir), filename=f"{paper_id}.pdf"))
        
        # 验证下载的文件
        if not output_path.exists():
            raise ProcessingError(f"PDF file was not saved properly for {paper_id}")
            
        file_size = output_path.stat().st_size
        if file_size < 1000:
            logger.warning(f"Downloaded PDF file for {paper_id} is suspiciously small: {file_size} bytes")
            raise ProcessingError(f"Downloaded PDF is too small ({file_size} bytes)")
            
        logger.info(f"Successfully downloaded PDF for {paper_id} to {output_path} ({file_size} bytes)")
        return output_path, metadata
            
    except Exception as e:
        logger.error(f"Error fetching/downloading PDF for {paper_id}: {e}")
        # raise ProcessingError(f"Error fetching/downloading PDF: {e}") # Re-raising obscures original error sometimes
        return None, None # Return None for both path and metadata on error

def extract_md_from_pdf_sync(paper_data):
    """
    同步版本的PDF文本提取函数，适用于多进程处理
    返回一个元组 (paper_id, md_path) 或 (paper_id, None)
    
    paper_data: 元组 (paper_id, pdf_path, output_dir)
    """
    paper_id, pdf_path, output_dir = paper_data
    try:
        logger.info(f"[ProcessPool] Starting text extraction for {paper_id} from {pdf_path}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 验证PDF文件是否存在且可访问
        if not os.path.exists(pdf_path):
            logger.error(f"[ProcessPool] PDF file does not exist: {pdf_path}")
            return paper_id, None
        
        if not os.access(pdf_path, os.R_OK):
            logger.error(f"[ProcessPool] PDF file is not readable: {pdf_path}")
            return paper_id, None
            
        file_size = os.path.getsize(pdf_path)
        if file_size < 1000:  # 假设小于1KB的PDF文件可能是损坏的
            logger.warning(f"[ProcessPool] PDF file for {paper_id} is suspiciously small ({file_size} bytes)")
        
        # 提取文本
        extracted_text = None # Initialize
        try:
            logger.info(f"[ProcessPool] Calling extract_text_from_pdf for {paper_id}")
            extracted_text = extract_text_from_pdf(pdf_path)
            if not extracted_text:
                logger.error(f"[ProcessPool] extract_text_from_pdf returned empty for {paper_id}")
                return paper_id, None
            logger.info(f"[ProcessPool] extract_text_from_pdf successful for {paper_id}, length: {len(extracted_text)}")
        except Exception as e:
            # !! 添加更详细的内部错误日志 !!
            logger.error(f"[ProcessPool] Exception INSIDE extract_text_from_pdf call for {paper_id}: {type(e).__name__} - {str(e)}")
            logger.exception("[ProcessPool] Full traceback for extract_text_from_pdf error:") # Log full traceback
            return paper_id, None
        
        # 保存提取的文本到临时MD文件
        md_path = output_dir / f"{paper_id}_extracted.md"
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            logger.info(f"[ProcessPool] Successfully wrote extracted text to {md_path}")
        except Exception as e:
            logger.error(f"[ProcessPool] Failed to write extracted text to file {md_path} for {paper_id}: {str(e)}")
            return paper_id, None
        
        logger.info(f"[ProcessPool] Completed text extraction for {paper_id}")
        return paper_id, str(md_path)
    except Exception as e:
        logger.error(f"[ProcessPool] Outer error during text extraction for {paper_id}: {e}")
        logger.exception("[ProcessPool] Full traceback for outer extraction error:") # Log full traceback
        return paper_id, None

async def extract_md_from_pdf(paper_id: str, pdf_path: Path, output_dir: Path) -> Path:
    """Extract Markdown text from a PDF file."""
    try:
        logger.info(f"Extracting text from PDF for {paper_id}")

        # 提取文本
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            # !! Log specific case of empty extraction !!
            logger.error(f"extract_text_from_pdf returned empty or None for {paper_id} at path {pdf_path}")
            raise ProcessingError(f"Failed to extract text from PDF for {paper_id} (empty result)")

        # 保存提取的文本到临时MD文件
        md_path = output_dir / f"{paper_id}_extracted.md"
        save_file(md_path, extracted_text, "Extracted Markdown")

        logger.info(f"Successfully extracted text for {paper_id}")
        return md_path
    except Exception as e:
        logger.error(f"Error extracting text from PDF for {paper_id}: {e}")
        raise ProcessingError(f"Error extracting text: {e}")

async def generate_summary(
    paper_id: str,
    md_path: Path,
    paper_metadata: dict, # Contains path, metadata, and score
    llm_client: LLMClient,
    model_name: str, # Add model name parameter
    base_prompt_instructions: str,
    renderer: TemplateRenderer,
    pdf_path: Path,
    output_dir: Path,
    stream_enabled: bool,
    display_lock: asyncio.Lock
) -> Path | None:
    """Generate a summary for a paper using the LLM."""
    try:
        logger.info(f"Generating summary for {paper_id} using model {model_name}")
        
        # 读取提取的MD文本
        async with aiofiles.open(md_path, 'r', encoding='utf-8') as f:
            extracted_text = await f.read()
        
        # 构造完整的prompt
        full_prompt = construct_full_prompt(base_prompt_instructions, extracted_text)
        
        # 调用LLM API生成摘要
        if stream_enabled:
            full_response = ""
            async with display_lock:
                print(f"\nGenerating summary for {paper_id}...")
            async for chunk in llm_client.get_summary_stream(full_prompt):
                full_response += chunk
                async with display_lock:
                    print(chunk, end="", flush=True)
            async with display_lock:
                print("\n")  # End the streaming display
        else:
            full_response = await llm_client.get_summary_no_stream(full_prompt)
        
        # 解析JSON响应
        try:
            summary_data = extract_summary_json(full_response)
        except SummaryParseError as e:
            logger.error(f"Failed to parse summary for {paper_id}: {e}")
            # Save error response to the *base* output dir for easier debugging
            error_path = output_dir / f"{paper_id}_error_response.txt"
            save_file(error_path, full_response, "Error Response")
            raise ProcessingError(f"Failed to parse summary JSON for {paper_id}: {e}")
        
        # 准备模板渲染所需的数据
        render_data = summary_data.copy()
        
        # Use metadata title (more reliable than LLM for title)
        render_data['paper_title'] = paper_metadata.get('metadata', {}).get('title', 'Unknown Title')
        render_data['paper_type'] = paper_metadata.get('type', 'Unknown Type')
        render_data['paper_id'] = paper_metadata.get('id', "Unknown ID")
        logger.debug(f"Using title for {paper_id}: {render_data['paper_title']}")

        # Date and time
        published_date = paper_metadata.get('metadata', {}).get('published_date')
        if published_date:
            render_data['pub_year'] = published_date.year
            render_data['pub_month'] = f"{published_date.month:02d}"
            render_data['time'] = published_date.isoformat()
        else:
            now = datetime.datetime.now()
            render_data['pub_year'] = now.year 
            render_data['pub_month'] = f"{now.month:02d}"
            render_data['time'] = now.isoformat()
            logger.warning(f"Missing published_date for {paper_id}, using current date.")
            
        # Score
        render_data['score'] = paper_metadata.get('score', 0)
        render_data['model_name'] = model_name # Add the model name here

        # Slug: Use LLM slug directly for template, generate fallback if needed
        llm_slug_base = render_data.get('slug') # Get slug from LLM JSON
        if not llm_slug_base:
             logger.error(f"LLM response for {paper_id} missing 'slug' field.")
             # Generate fallback base slug from the reliable title
             llm_slug_base = re.sub(r'\W+', '-', render_data['paper_title'].lower()).strip('-')[:50]
             render_data['slug'] = llm_slug_base # Store fallback in render_data for template
             logger.warning(f"Generated fallback slug for {paper_id}: {llm_slug_base}")
        else:
            render_data['slug'] = llm_slug_base # Ensure the key exists with the LLM value
            logger.info(f"Using LLM provided slug base for {paper_id}: {llm_slug_base}")

        # Construct the full slug for filename generation (outside render_data)
        filename_slug = f"{render_data['pub_year']}-{render_data['pub_month']}-{llm_slug_base}"
        
        # 添加日志：确认最终渲染的标题
        final_title_for_render = render_data.get('paper_title', 'TITLE MISSING IN RENDER_DATA')
        logger.info(f"Rendering template for {paper_id} with FINAL title: '{final_title_for_render}' and slug base: '{render_data['slug']}'")

        # 渲染摘要为markdown
        rendered_md = renderer.render_summary(render_data, original_pdf_path=pdf_path)
        
        # 构建最终的输出路径
        final_output_dir = output_dir / str(render_data['pub_year']) / str(render_data['pub_month'])
        # 确保目标目录存在
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存最终的markdown到目标目录 (使用 filename_slug)
        output_filename = f"{filename_slug}.md"
        output_path = final_output_dir / output_filename
        save_file(output_path, rendered_md, "Final Markdown")
        
        logger.info(f"Successfully generated summary for {paper_id} saved to {output_path}")
        return output_path
    except ProcessingError as e:
        logger.error(f"Processing error generating summary for {paper_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating summary for {paper_id}: {e}")
        logger.exception("Full exception details:")
        return None

def pdf_processor_worker(worker_id, pdf_queue, extraction_results, extraction_lock):
    """
    进程池工作线程：从队列获取PDF信息并提取文本
    """
    logger.info(f"PDF processor worker {worker_id} started")
    
    while True:
        paper_id = None # Initialize paper_id for logging in case queue.get fails
        task_data = None
        try:
            # 尝试从队列获取任务，阻塞等待，不设置超时
            # If get() returns None, it's the sentinel value
            task_data = pdf_queue.get() 
            
            # Check for sentinel value to terminate worker
            if task_data is None:
                logger.info(f"Worker {worker_id} received sentinel None. Exiting loop.")
                break # Exit the loop

            # If not None, unpack the task data
            paper_id, pdf_path, output_dir = task_data
            logger.info(f"Worker {worker_id} received task for {paper_id}")


            logger.info(f"Worker {worker_id} processing PDF for {paper_id} from path {pdf_path}")

            # 执行文本提取
            result = extract_md_from_pdf_sync((paper_id, pdf_path, output_dir))
            # !! Unpack result carefully !!
            res_paper_id, md_path = result # Use different variable name to avoid confusion

            # 更新结果
            with extraction_lock:
                if md_path:
                    extraction_results[res_paper_id] = str(md_path) # Ensure result is string path
                    logger.info(f"Worker {worker_id} successfully extracted {res_paper_id} to {md_path}. Updated shared results.")
                else:
                    # Log failure, but don't add to extraction_results
                    logger.warning(f"Worker {worker_id} failed extraction for {res_paper_id}. Not adding to results.")

        except Exception as e:
            # Log error related to this specific paper_id if available
            log_msg = f"Error in worker {worker_id}"
            if paper_id:
                log_msg += f" while processing paper {paper_id}"
            log_msg += f": {e}"
            logger.error(log_msg)
            logger.exception(f"Full traceback for error in worker {worker_id}:") # Log traceback

    logger.info(f"PDF processor worker {worker_id} finished")

async def pipeline_process_pdfs(temp_dir, max_workers):
    """使用生产者-消费者模式并行处理PDF提取"""
    manager = multiprocessing.Manager()
    pdf_queue = manager.Queue()
    extraction_results = manager.dict()
    extraction_lock = manager.Lock()
    
    # 启动消费者进程池
    processes = []
    for i in range(max_workers):
        p = multiprocessing.Process(
            target=pdf_processor_worker, 
            args=(i, pdf_queue, extraction_results, extraction_lock)
        )
        p.daemon = True  # 设置为守护进程，这样在主进程退出时会自动终止
        p.start()
        processes.append(p)
    
    logger.info(f"Started {len(processes)} PDF processor workers")
    
    # 返回共享对象和进程列表，以便主进程可以向队列添加任务和等待
    return pdf_queue, extraction_results, extraction_lock, processes

async def wait_for_processors(processes, extraction_results, timeout=300):
    """等待所有处理进程完成"""
    logger.info(f"Waiting for {len(processes)} worker processes to finish...")
    # 等待所有工作进程完成
    for p in processes:
        logger.debug(f"Waiting for process {p.pid} (daemon: {p.daemon})...")
        p.join(timeout=timeout)  # 设置超时时间，防止无限等待
        if p.is_alive():
            logger.warning(f"Worker process {p.pid} did not terminate after {timeout}s, force terminating")
            p.terminate() # Force terminate if timed out
            p.join(5) # Wait briefly for termination
            if p.is_alive():
                 logger.error(f"Failed to terminate worker process {p.pid}")
        else:
            logger.info(f"Worker process {p.pid} finished with exit code {p.exitcode}")

    # 转换结果为普通字典
    # !! Critical: Access manager dict *after* confirming processes joined/terminated !!
    logger.info("All worker processes joined or terminated. Accessing extraction results...")
    try:
        # Create a copy to avoid issues if manager shuts down
        results_copy = dict(extraction_results)
        md_paths = {k: Path(v) for k, v in results_copy.items()} # Convert string paths back to Path objects
        logger.info(f"Successfully retrieved {len(md_paths)} results from workers.")
    except Exception as e:
        logger.error(f"Error accessing results from manager dict: {e}")
        logger.exception("Full traceback for manager dict access error:")
        md_paths = {} # Return empty dict on error


    logger.info(f"Pipeline processing completed, extracted {len(md_paths)} PDFs")
    return md_paths

async def main():
    # Load default config first
    default_config_path = DEFAULT_CONFIG_PATH
    if default_config_path.exists():
        config = load_config(default_config_path)
    else:
        logger.warning(f"Default config file not found at {default_config_path}, using hardcoded defaults")
        config = {}
    
    # Extract defaults from config (from summary section)
    summary_config = config.get('summary', {})
    
    # Default values from config or hardcoded if not in config
    csv_path_default = summary_config.get('csv_path', '')
    output_dir_default = summary_config.get('output_dir', str(DEFAULT_OUTPUT_DIR))
    max_workers_default = summary_config.get('max_workers', summary_config.get('max_concurrent_llm_calls', 3))
    stream_enabled_default = summary_config.get('stream', False)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate AI summaries for arXiv papers.")
    parser.add_argument(
        "csv_path", 
        type=str, 
        nargs='?',  # Make it optional
        default=csv_path_default,
        help="Path to the CSV file containing arXiv paper information."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to configuration file. Default: {DEFAULT_CONFIG_PATH}"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=output_dir_default,
        help=f"Directory to save output markdown files. Default from config or {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=max_workers_default,
        help=f"Maximum number of concurrent workers. Default from config or {max_workers_default}"
    )
    parser.add_argument(
        "--max-process-workers",
        type=int,
        default=None,  # 默认使用CPU核心数
        help="Maximum number of process workers for PDF processing. Default: CPU core count"
    )
    parser.add_argument(
        "--stream", 
        action="store_true",
        default=stream_enabled_default,
        help="Enable streaming mode for LLM responses."
    )
    
    args = parser.parse_args()
    
    # 设置多进程启动方法（Windows上使用'spawn'）
    multiprocessing.set_start_method('spawn', force=True)
    
    # 设置多进程工作数量，如果未指定则使用CPU核心数
    if args.max_process_workers is None:
        # 使用CPU核心数（减1，保留一个核心给主线程）
        args.max_process_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
    
    logger.info(f"Using {args.max_process_workers} process workers for PDF processing")
    logger.info(f"Using {args.max_workers} async workers for downloads and LLM calls")
    
    # Check if config path from args is different from default
    if Path(args.config) != default_config_path:
        # Reload config if a different file was specified
        config = load_config(Path(args.config))
        # Re-extract defaults for any values that weren't specified on command line
        summary_config = config.get('summary', {})
    
    # Ensure we have a CSV path from either command line or config
    if not args.csv_path:
        logger.error("No CSV file path provided via command line or config file")
        sys.exit(1)
    
    # Convert string paths to Path objects
    csv_path = Path(args.csv_path)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    # 创建一个本地目录用于保存临时文件，放在脚本同目录下而不是系统临时目录
    script_dir = Path(__file__).parent
    temp_dir_name = f"arxiv_downloads_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir = script_dir / temp_dir_name
    logger.info(f"Creating local directory for downloads: {temp_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure CSV file exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.datetime.now()
    
    # 加载CSV数据并过滤
    papers_map = {} # Store original paper data by id
    try:
        logger.info(f"Reading CSV file: {csv_path}")
        # Specify types for id (string) and score (float or int)
        # Use Float64 for score to handle potential decimals and nulls gracefully
        df = pl.read_csv(csv_path, schema_overrides={"id": pl.Utf8, "score": pl.Float64})
        
        logger.debug(f"CSV schema: {df.schema}")
        
        # Filter rows
        try:
            filtered_df = df.filter(
                (pl.col("type") == "arxiv") & 
                (pl.col("show") == 1)
            )
            # get first 10 rows
            filtered_df = filtered_df.head(3)
        except Exception as filter_error:
            logger.error(f"Error filtering CSV data: {filter_error}")
            logger.error("Ensure CSV has 'type' and 'show' columns.")
            sys.exit(1)
        
        # Check for required columns
        required_columns = ["id", "pdf_url", "score"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"CSV file missing required column: '{col}'")
                sys.exit(1)
        
        # Extract paper info into a map for easy lookup
        for row in filtered_df.iter_rows(named=True):
            paper_id = str(row["id"])
            papers_map[paper_id] = {
                "id": paper_id,
                "pdf_url": row["pdf_url"],
                "score": row["score"], # Score is now included
                "type": row["type"],
                "id": row["id"]
            }
        
        if not papers_map:
            logger.warning("No arXiv papers to process after filtering.")
            sys.exit(0)
            
        logger.info(f"Found {len(papers_map)} arXiv papers to process")
        
    except Exception as e:
        logger.error(f"Failed to process CSV file: {e}")
        logger.exception("Full exception details:")
        sys.exit(1)
    
    # 加载prompt指令
    prompt_path = Path(summary_config.get("prompt_path", config_path.parent / "prompt.txt"))
    base_prompt_instructions = load_prompt_instructions(prompt_path)
    
    # 初始化模板渲染器
    template_path = Path(summary_config.get("template_path", config_path.parent / "template.j2"))
    try:
        renderer = TemplateRenderer(str(template_path))
    except TemplateError as e:
        logger.error(f"Failed to initialize template renderer: {e}")
        sys.exit(1)
    
    # 初始化LLM客户端
    try:
        llm_client = await init_llm_client(config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    # 并发控制器
    semaphore = asyncio.Semaphore(args.max_workers)
    display_lock = asyncio.Lock()

    # 阶段1+2: 下载PDF并实时处理
    logger.info("=== 阶段1+2: 开始下载PDF并实时处理 ===")
    pdf_data_store = {}  # Store path, metadata, and score per paper_id
    
    # 初始化处理队列和工作进程
    pdf_queue, extraction_results, extraction_lock, processors = await pipeline_process_pdfs(temp_dir, args.max_process_workers)
    logger.info("PDF processing pipeline initialized")
    
    queued_pdfs = 0
    
    async with aiohttp.ClientSession() as session:
        download_tasks = []
        for paper_id, paper_info in papers_map.items(): # Iterate through the map
            pdf_url = paper_info["pdf_url"]
            paper_dir = temp_dir / paper_id
            
            async def download_with_semaphore(paper_id, pdf_url, paper_dir):
                async with semaphore:
                    pdf_path, metadata = await download_pdf(session, paper_id, pdf_url, paper_dir)
                    return paper_id, pdf_path, metadata
            
            task = download_with_semaphore(paper_id, pdf_url, paper_dir)
            download_tasks.append(task)
        
        # 处理下载结果
        for download_task in asyncio.as_completed(download_tasks):
            paper_id, pdf_path, metadata = await download_task
            if pdf_path and metadata:
                # Successfully downloaded, now add score from original map
                original_score = papers_map.get(paper_id, {}).get('score', 0) # Get score with fallback
                pdf_data_store[paper_id] = {
                    "path": pdf_path,
                    "metadata": metadata,
                    "score": original_score, # Add the score here
                    "type": papers_map.get(paper_id, {}).get('type', 'Unknown Type'),
                    "id": paper_id
                }
                # 加入处理队列
                paper_dir = temp_dir / paper_id
                # !! Ensure pdf_path is serializable (string) for the queue !!
                pdf_queue.put((paper_id, str(pdf_path), paper_dir))
                queued_pdfs += 1
                logger.info(f"Queued PDF for {paper_id} (#{queued_pdfs}) with score: {original_score}")
            else:
                 logger.error(f"Download or metadata fetch failed for {paper_id}. Skipping.")
    
    # 检查下载结果
    if not pdf_data_store:
        logger.error("No PDFs were successfully downloaded with metadata. Exiting.")
        for p in processors:
            p.terminate()
        sys.exit(1)
    
    logger.info(f"Successfully fetched {len(pdf_data_store)} PDFs with metadata, added {queued_pdfs} to extraction queue") # Log how many were queued
    
    # === Signal workers to finish ===
    # After all tasks are queued, put one sentinel (None) for each worker process
    logger.info(f"All download tasks completed. Adding {args.max_process_workers} sentinel values to the queue.")
    for _ in range(args.max_process_workers):
        pdf_queue.put(None)
    logger.info("Sentinels added to the queue.")

    # 等待所有文本提取完成
    logger.info("Waiting for all PDF extraction tasks to complete...")
    md_paths = await wait_for_processors(processors, extraction_results)
    
    # 检查提取结果
    if not md_paths:
        logger.error("No text was successfully extracted from PDFs (md_paths is empty). Exiting.")
        # !! Add check for discrepancy !!
        if queued_pdfs > 0:
             logger.warning(f"{queued_pdfs} PDFs were queued for extraction, but none succeeded.")
        sys.exit(1)
    
    logger.info(f"Extracted text from {len(md_paths)} PDFs successfully.")
    
    # 阶段3: 批量调用LLM生成摘要
    logger.info("=== 阶段3: 开始批量生成摘要 ===")
    
    summary_tasks = []
    successful_papers = 0
    failed_papers = 0
    
    for paper_id, md_path in md_paths.items():
        if paper_id in pdf_data_store:
            # Pass the whole dict containing path, metadata, and score
            paper_full_data = pdf_data_store[paper_id]
            pdf_path = paper_full_data["path"] # Needed for generate_summary signature? Refactor maybe
            
            # Get model name from config
            model_name = summary_config.get('model', 'unknown_model')
            logger.info(f"Using LLM model: {model_name}")
            
            async def generate_with_semaphore(paper_id, md_path, paper_full_data, pdf_path, model_name):
                async with semaphore:
                    result_path = await generate_summary(
                        paper_id,
                        md_path,
                        paper_full_data,
                        llm_client,
                        model_name,
                        base_prompt_instructions,
                        renderer,
                        pdf_path,
                        output_dir,
                        args.stream,
                        display_lock
                    )
                    return result_path is not None
            
            task = generate_with_semaphore(paper_id, md_path, paper_full_data, pdf_path, model_name)
            summary_tasks.append(task)
        else:
             logger.warning(f"Metadata/Score missing for successfully extracted paper {paper_id}. Skipping summary generation.")
             failed_papers += 1

    # 等待所有摘要生成任务完成
    summary_results = await asyncio.gather(*summary_tasks)
    
    # 计算成功和失败数量
    successful_summaries = sum(1 for success in summary_results if success)
    failed_summaries = len(summary_results) - successful_summaries + failed_papers
    total_processed = len(md_paths)

    # 记录结束时间和总耗时
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Completed processing {len(papers_map)} initial papers in {duration.total_seconds():.2f} seconds")
    logger.info(f"Summary Generation: {successful_summaries} succeeded, {failed_summaries} failed (out of {total_processed} attempted)")
    
    # 清理工作
    await llm_client.close()
    
    try:
        if temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            logger.info("Removed temporary directory")
    except Exception as e:
        logger.warning(f"Failed to remove temporary directory: {e}")
    
    if failed_summaries > 0:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
