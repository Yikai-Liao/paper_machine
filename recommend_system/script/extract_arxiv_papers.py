import argparse
import os
import pathlib
import sys
import time
import re
import random
import polars as pl
from loguru import logger
import arxiv
import pytz
from datetime import datetime

# --- 设置日志 ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- Helper Function for ID Normalization ---
def normalize_arxiv_id(raw_id: str) -> str:
    """Normalizes arXiv ID by removing version and padding the suffix with zeros."""
    if not isinstance(raw_id, str):
        # Attempt conversion if not string, log warning
        logger.warning(f"Attempting to normalize non-string ID: {raw_id} of type {type(raw_id)}. Converting to string.")
        raw_id = str(raw_id)
    
    # 如果是完整URL，提取ID部分
    if raw_id.startswith("http") and "arxiv.org" in raw_id:
        raw_id = raw_id.split('/')[-1]
    
    # 移除版本号
    base_id = re.sub(r'v\d+$', '', raw_id.strip())
    
    # 处理常见的arXiv ID格式
    if "arxiv:" in base_id.lower():
        base_id = base_id.lower().replace("arxiv:", "")
    
    parts = base_id.split('.')
    if len(parts) == 2:
        year_month = parts[0]
        suffix = parts[1]
        
        # 检查常见格式YYMM或YYYY.MM
        if not (len(year_month) == 4 or (len(year_month) == 7 and year_month[4] == '.')):
             logger.warning(f"Unexpected format before dot in ID {raw_id}. Returning base ID {base_id}.")
             return base_id
        
        try:
            # 将小数点后部分规范为5位数，前面补0
            normalized_suffix = f"{int(suffix):05d}"
            normalized_id = f"{year_month}.{normalized_suffix}"
            return normalized_id
        except ValueError:
            logger.warning(f"Could not normalize numeric suffix for ID {raw_id}. Returning base ID {base_id}.")
            return base_id
    
    return base_id

# --- 从paperlib CSV提取arXiv ID ---
def extract_arxiv_ids_from_paperlib(csv_path):
    """从paperlib CSV文件提取arXiv ID"""
    logger.info(f"读取paperlib CSV文件: {csv_path}")
    try:
        # 使用polars读取CSV文件
        df = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8, 'arxiv': pl.Utf8})
        
        # 检查是否包含arxiv列
        if 'arxiv' in df.columns:
            arxiv_ids = df.filter(pl.col('arxiv').is_not_null())['arxiv'].to_list()
            logger.info(f"从arxiv列中提取到 {len(arxiv_ids)} 个arXiv ID")
        else:
            logger.warning("CSV文件中没有找到arxiv列，尝试从其他列中提取arXiv ID")
            # 如果没有arxiv列，尝试从其他可能的列中提取
            potential_columns = ['doi', 'mainURL', 'note']
            arxiv_ids = []
            
            for col in potential_columns:
                if col in df.columns:
                    # 尝试从列中提取arXiv ID
                    potential_ids = []
                    for value in df[col].to_list():
                        if value and isinstance(value, str):
                            # 尝试匹配arXiv ID格式
                            if "arxiv" in value.lower() or re.search(r'\d{4}\.\d{4,5}', value):
                                potential_ids.append(value)
                    
                    if potential_ids:
                        logger.info(f"从{col}列中找到 {len(potential_ids)} 个潜在的arXiv ID")
                        arxiv_ids.extend(potential_ids)
        
        # 归一化arXiv ID
        normalized_ids = [normalize_arxiv_id(arxiv_id) for arxiv_id in arxiv_ids if arxiv_id]
        unique_ids = list(set(normalized_ids))  # 去重
        
        logger.info(f"提取并归一化了 {len(unique_ids)} 个唯一的arXiv ID")
        return unique_ids
        
    except Exception as e:
        logger.exception(f"读取CSV文件时发生错误: {e}")
        return []

# --- 使用arXiv API获取论文详情 ---
def fetch_paper_details(arxiv_ids, delay_seconds=3.0, max_batch_size=100):
    """使用arXiv API获取论文详情"""
    if not arxiv_ids:
        logger.warning("没有提供arXiv ID，无法获取论文详情")
        return []
    
    logger.info(f"开始获取 {len(arxiv_ids)} 篇论文的详情")
    
    # 创建arXiv客户端
    client = arxiv.Client(page_size=max_batch_size, delay_seconds=delay_seconds)
    
    # 分批获取详情以避免API限制
    papers_details = []
    batch_size = min(max_batch_size, len(arxiv_ids))
    
    for i in range(0, len(arxiv_ids), batch_size):
        batch_ids = arxiv_ids[i:i+batch_size]
        logger.info(f"获取第 {i//batch_size + 1} 批论文 ({len(batch_ids)} 篇)")
        
        try:
            # 创建搜索查询
            search = arxiv.Search(id_list=batch_ids, max_results=len(batch_ids))
            batch_results = list(client.results(search))
            
            for result in batch_results:
                arxiv_id_raw = result.entry_id.split('/')[-1]
                normalized_id = normalize_arxiv_id(arxiv_id_raw)
                
                # 格式化作者
                authors = '; '.join([author.name for author in result.authors])
                
                # 处理日期，确保包含时区
                published_date = result.published
                if published_date and published_date.tzinfo is None:
                    published_date = pytz.utc.localize(published_date)
                published_date_iso = published_date.isoformat() if published_date else None
                
                # 整理论文信息
                paper_details = {
                    'type': 'arxiv',
                    'id': normalized_id,
                    'title': result.title.replace('\n', ' ').strip(),
                    'authors': authors,
                    'abstract': result.summary.replace('\n', ' ').strip() if result.summary else "",
                    'date': published_date_iso,
                    'primary_category': result.primary_category,
                    'pdf_url': result.pdf_url,
                    'preference': 'like',  # 添加preference列，值为"like"
                    'score': round(random.uniform(0.05, 0.7), 16),  # 添加随机score
                    'show': 0  # 添加show列，值为0
                }
                
                papers_details.append(paper_details)
            
            logger.info(f"已获取 {len(batch_results)}/{len(batch_ids)} 篇论文的详情")
            
            # 添加延迟，避免过快请求API
            if i + batch_size < len(arxiv_ids):
                logger.info(f"等待 {delay_seconds} 秒后获取下一批...")
                time.sleep(delay_seconds)
                
        except Exception as e:
            logger.exception(f"获取论文详情时发生错误: {e}")
    
    logger.info(f"总共获取了 {len(papers_details)} 篇论文的详情")
    return papers_details

# --- 保存为arxiv_latest.csv格式 ---
def save_to_arxiv_format(papers_details, output_path):
    """将论文详情保存为arxiv_latest.csv格式"""
    if not papers_details:
        logger.warning("没有论文详情可保存")
        return False
    
    try:
        # 创建DataFrame
        df = pl.DataFrame(papers_details)
        
        # 确保列顺序与arxiv_latest.csv一致
        column_order = [
            'type', 'id', 'title', 'authors', 'date', 'primary_category', 
            'pdf_url', 'abstract', 'score', 'show', 'preference'
        ]
        
        # 检查所有必要的列是否存在，如果不存在则添加
        for col in column_order:
            if col not in df.columns:
                if col in ['score', 'show']:
                    df = df.with_columns(pl.lit(0).alias(col))
                elif col == 'preference':
                    df = df.with_columns(pl.lit('like').alias(col))
                else:
                    df = df.with_columns(pl.lit("").alias(col))
        
        # 重排列顺序并保存
        df = df.select(column_order)
        df.write_csv(output_path)
        
        logger.info(f"已将 {len(papers_details)} 篇论文的详情保存到 {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"保存论文详情时发生错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="从paperlib CSV提取arXiv论文并转换为arxiv_latest.csv格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入的paperlib CSV文件路径")
    parser.add_argument("--output", "-o", type=str, default="data/arxiv_from_paperlib.csv",
                        help="输出的arxiv格式CSV文件路径")
    parser.add_argument("--delay", "-d", type=float, default=3.0,
                        help="arXiv API请求之间的延迟时间（秒）")
    
    args = parser.parse_args()
    
    # 确保输入文件存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 从paperlib CSV提取arXiv ID
    arxiv_ids = extract_arxiv_ids_from_paperlib(args.input)
    
    if not arxiv_ids:
        logger.error("没有找到任何arXiv ID，程序终止")
        return
    
    # 2. 获取论文详情
    papers_details = fetch_paper_details(arxiv_ids, args.delay)
    
    if not papers_details:
        logger.error("无法获取任何论文详情，程序终止")
        return
    
    # 3. 保存为arxiv_latest.csv格式
    success = save_to_arxiv_format(papers_details, args.output)
    
    if success:
        logger.info("转换完成！")
    else:
        logger.error("转换失败")

if __name__ == "__main__":
    main() 