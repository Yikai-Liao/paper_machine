import argparse
import os
import pathlib
import sys
import toml
import numpy as np
import polars as pl
from loguru import logger

# --- 设置相对路径导入 ---
# 获取当前脚本所在的目录
script_dir = pathlib.Path(__file__).parent.resolve()
# 获取项目根目录 (script 目录的上级目录)
project_root = script_dir.parent
# 将项目根目录添加到 sys.path
sys.path.insert(0, str(project_root))

# 现在可以进行相对导入
try:
    from src.embed import AliCloudEmbed, EmbedBase # Import the specific embedder
except ImportError as e:
    logger.error(f"Failed to import embedding module: {e}")
    logger.error("Ensure the script is run from the correct location or sys.path is configured.")
    sys.exit(1)

# --- 日志配置 ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 配置加载 ---
def load_config() -> dict:
    """Loads configuration from config.toml located in the script's parent directory.
       Looks for [embedding] section. Returns the section content or empty dict.
    """
    config_path = project_root / "config.toml"
    config_section = {}
    if config_path.is_file():
        logger.info(f"Loading configuration from: {config_path}")
        try:
            loaded_config = toml.load(config_path)
            config_section = loaded_config.get('embedding', {})
            logger.info(f"Loaded config from [embedding] section: {config_section}")
        except toml.TomlDecodeError as e:
            logger.exception(f"Error decoding config file {config_path}: {e}")
        except Exception as e:
            logger.exception(f"Error loading config file {config_path}: {e}")
    else:
        logger.warning(f"Configuration file not found at {config_path}. Embedding requires configuration (e.g., api_key).")
    return config_section

# --- 主函数 ---
def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Generate embeddings for abstracts in a CSV file using AliCloudEmbed. Reads config from ../config.toml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-csv", type=str,
                        default=config.get('input_csv'),
                        help="Path to the input CSV file containing abstracts.")
    parser.add_argument("--output-npy", type=str,
                        default=config.get('output_npy'),
                        help="Path to save the output NumPy (.npy) file. If omitted, derived from input CSV path.")
    parser.add_argument("--abstract-col", type=str, default="abstract",
                        help="Name of the column containing abstracts in the CSV.")

    args = parser.parse_args()

    # --- 参数校验和路径处理 ---
    input_csv_path_str = args.input_csv
    output_npy_path_str = args.output_npy
    abstract_col = args.abstract_col

    if not input_csv_path_str:
        logger.error("Input CSV path is required. Provide via --input-csv or config.toml [embedding].input_csv")
        parser.print_help()
        return

    input_csv_path = pathlib.Path(input_csv_path_str).resolve()
    if not input_csv_path.is_file():
        logger.error(f"Input CSV file not found: {input_csv_path}")
        return

    # Derive output path if not provided
    if not output_npy_path_str:
        output_npy_path = input_csv_path.with_suffix('.npy')
    else:
        # Ensure output path has .npy suffix if specified
        output_npy_path = pathlib.Path(output_npy_path_str).resolve()
        if output_npy_path.suffix != '.npy':
             logger.warning(f"Output path {output_npy_path} does not end with .npy. Appending suffix.")
             output_npy_path = output_npy_path.with_suffix('.npy')


    logger.info(f"Input CSV: {input_csv_path}")
    logger.info(f"Output NPY: {output_npy_path}")
    logger.info(f"Abstract column: {abstract_col}")

    # --- 读取 CSV 数据 ---
    try:
        logger.info("Reading CSV file...")
        df = pl.read_csv(input_csv_path)
        
        # 检查abstract列
        if abstract_col not in df.columns:
            logger.error(f"Column '{abstract_col}' not found in {input_csv_path}. Available columns: {df.columns}")
            return
            
        # 检查title列
        if 'title' not in df.columns:
            logger.warning(f"Column 'title' not found in {input_csv_path}. Using empty titles in embedding format.")
            df = df.with_columns(pl.lit("").alias("title"))
        
        # 过滤掉摘要为空的行
        df = df.filter(pl.col(abstract_col).is_not_null())
        
        if df.is_empty():
            logger.warning(f"No non-null abstracts found in column '{abstract_col}'. Exiting.")
            return
             
        # 只使用同时有标题和摘要的行
        df_for_embed = df.filter(
            pl.col('title').is_not_null() & 
            pl.col('abstract').is_not_null()
        )
        
        if df_for_embed.height == 0:
            logger.warning("No rows with both title and abstract found. Cannot generate embeddings.")
            return None
        
        # 使用select获取列，然后用rows()方法获取行数据
        rows_data = df_for_embed.select([
            pl.col('title'),
            pl.col('abstract')
        ]).rows()
        
        # 使用列表推导式创建嵌入文本
        embedding_texts = [f"Title: {row[0]}\nAbstract: {row[1]}" for row in rows_data]
        
        logger.info(f"Prepared {len(embedding_texts)} documents with 'Title: {{}}\nAbstract: {{}}' format for embedding.")

    except Exception as e:
        logger.exception(f"Failed to read or process CSV file: {input_csv_path}")
        return

    # --- 初始化 Embedding 模型 ---
    # Get API key from config, check if it's set to "env"
    api_key_config = config.get('api_key')
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
        logger.error("AliCloud API key is missing. Provide it in config.toml [embedding].api_key or set it to 'env' and export EMBEDDING_API_KEY.")
        return

    # api_key should now hold the actual key or the script would have exited
    base_url = config.get('base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = config.get('model', "text-embedding-v3")
    dim = config.get('dim', 1024)

    try:
        logger.info(f"Initializing AliCloudEmbed model: {model} (dim: {dim}) at endpoint: {base_url}")
        embedder: EmbedBase = AliCloudEmbed(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dim=dim
        )
    except Exception as e:
        logger.exception("Failed to initialize embedding model.")
        return

    # --- 生成 Embeddings ---
    try:
        logger.info("Generating embeddings... (This may take a while for large datasets)")
        embeddings = embedder.embed(embedding_texts) # 使用新的格式化文本
        logger.info(f"Successfully generated embeddings. Shape: {embeddings.shape}")

    except Exception as e:
        logger.exception("Failed to generate embeddings.")
        return
        
    if embeddings.shape[0] != len(embedding_texts):
         logger.warning(f"Number of embeddings ({embeddings.shape[0]}) does not match number of input texts ({len(embedding_texts)}). Check batch API results.")
         # Decide whether to proceed or stop based on policy
         # return

    # --- 保存 Embeddings ---
    try:
        logger.info(f"Saving embeddings to {output_npy_path}...")
        # Ensure output directory exists
        output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为fp16精度再保存
        embeddings_fp16 = embeddings.astype(np.float16)
        original_size = embeddings.nbytes / (1024 * 1024)
        fp16_size = embeddings_fp16.nbytes / (1024 * 1024)
        logger.info(f"Converting embeddings to fp16 precision. Original size: {original_size:.2f}MB, fp16 size: {fp16_size:.2f}MB")
        
        # Save as uncompressed numpy array with fp16 precision
        np.save(str(output_npy_path), embeddings_fp16, allow_pickle=False)
        logger.info("Embeddings saved successfully in fp16 precision.")
    except Exception as e:
        logger.exception(f"Failed to save embeddings to {output_npy_path}")

if __name__ == "__main__":
    main()
