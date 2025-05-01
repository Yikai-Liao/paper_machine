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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import platform

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

# --- 配置matplotlib中文字体支持 ---
def configure_chinese_font():
    """配置matplotlib支持中文字体"""
    if platform.system() == "Windows":
        # Windows系统优先使用微软雅黑
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun']
    elif platform.system() == "Darwin":
        # macOS系统
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        # Linux或其他系统
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC']
    
    # 尝试设置中文字体
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            logger.info(f"设置matplotlib中文字体: {font}")
            # 测试字体是否成功设置
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
            plt.close(fig)  # 关闭测试图形
            return True
        except Exception as e:
            logger.debug(f"尝试设置字体{font}失败: {e}")
    
    # 如果所有字体都失败，使用特殊方法
    try:
        logger.warning("尝试使用matplotlib内部字体管理器解决中文显示问题")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        logger.error(f"配置中文字体失败: {e}")
        return False

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

# --- 新增绘图功能 ---
def plot_score_distributions(data_dict: Dict[str, np.ndarray], threshold: float = None, 
                            output_path: str = None) -> None:
    """绘制不同数据集的分数分布曲线
    
    Args:
        data_dict: 字典，键为数据集名称，值为预测分数数组
        threshold: 不再使用的参数，保留是为了兼容性
        output_path: 可选的输出文件路径
    """
    plt.figure(figsize=(12, 7))
    
    # 筛选要显示的三个分布：目标数据、背景数据、训练数据正样本
    filtered_data = {}
    keys_to_keep = ["目标数据", "背景数据", "训练数据(正样本)"]
    
    for key in keys_to_keep:
        if key in data_dict and len(data_dict[key]) > 0:
            # 确保数据在[0,1]范围内
            scores = np.clip(data_dict[key], 0, 1)
            filtered_data[key] = scores
    
    # 设置颜色映射
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 使用直方图和KDE组合
    for i, (name, scores) in enumerate(filtered_data.items()):
        if len(scores) > 0:
            # 绘制直方图
            plt.hist(scores, bins=30, alpha=0.3, density=True, 
                    color=colors[i % len(colors)], label=f'{name} (n={len(scores)})')
            
            # 叠加KDE曲线，减少平滑度
            sns.kdeplot(scores, color=colors[i % len(colors)], 
                       alpha=0.8, lw=2, bw_adjust=0.5)  # bw_adjust<1减少平滑
    
    # 只添加0.5固定阈值线
    plt.axvline(x=0.5, color='red', linestyle='--', 
               linewidth=2, label='阈值 = 0.5')
    
    # 设置x轴范围为[0,1]
    plt.xlim(0, 1)
    plt.title('分数分布对比', fontsize=15)
    plt.xlabel('预测分数', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存或显示图表
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"分数分布图已保存至: {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, 
                  beta: float, output_path: str = None) -> None:
    """绘制ROC曲线并标记最佳阈值点
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        threshold: 最佳阈值
        beta: 用于F-beta/G-mean的beta值
        output_path: 可选的输出文件路径
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 找到最接近threshold的点
    threshold_idx = np.argmin(np.abs(thresholds - threshold)) if len(thresholds) > 0 else 0
    
    plt.figure(figsize=(10, 8))
    
    # 使用更鲜明的颜色和更粗的线条
    plt.plot(fpr, tpr, color='blue', lw=2.5, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    
    # 增大标记点和改善标签可见性
    if threshold_idx < len(fpr):
        plt.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red', s=120, zorder=10,
                    label=f'阈值 = {threshold:.3f}\nFPR = {fpr[threshold_idx]:.3f}\nTPR = {tpr[threshold_idx]:.3f}')
    
    # 设置坐标轴范围和标签
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title(f'ROC曲线分析 (AUC = {roc_auc:.3f})', fontsize=15)
    
    # 添加网格线以提高可读性
    plt.grid(True, alpha=0.4, linestyle=':')
    
    # 优化图例位置和样式
    plt.legend(loc="lower right", fontsize=11, framealpha=0.8)
    
    # 保存或显示图表
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC曲线图已保存至: {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def load_and_combine_preference_data(pref_dir: pathlib.Path) -> Optional[pl.DataFrame]:
    """递归加载pref_dir中的CSV文件，查找匹配的NPY文件，并合并数据。
    自动过滤空ID和空embedding。
    """
    all_pref_dfs = []
    csv_files = list(pref_dir.rglob("*.csv"))
    logger.info(f"在{pref_dir}中找到{len(csv_files)}个潜在的CSV文件。")

    if not csv_files:
        logger.error(f"在偏好目录中没有找到CSV文件: {pref_dir}")
        return None

    loaded_ids = set()  # 跟踪已加载的ID，避免跨文件重复

    for csv_path in csv_files:
        npy_path = csv_path.with_suffix('.npy')
        logger.debug(f"处理CSV: {csv_path}")
        if not npy_path.is_file():
            logger.warning(f"没有找到匹配的NPY文件: {csv_path}。跳过此文件。")
            continue

        try:
            df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
            logger.debug(f"加载了CSV {csv_path}，形状 {df_csv.shape}")
            
            # 过滤空ID
            if df_csv["id"].is_null().any():
                null_count = df_csv["id"].is_null().sum()
                logger.warning(f"在{csv_path}中发现{null_count}个空ID，将被过滤掉")
                df_csv = df_csv.filter(pl.col("id").is_not_null())
                
                if df_csv.is_empty():
                    logger.warning(f"过滤空ID后，{csv_path}中没有剩余数据。跳过此文件。")
                    continue
                    
            required_cols = {'id', 'preference'}
            if not required_cols.issubset(df_csv.columns):
                logger.warning(f"CSV {csv_path} 缺少必要的列 ({required_cols})。跳过。")
                continue

            embeddings = np.load(npy_path)
            logger.debug(f"加载了NPY {npy_path}，形状 {embeddings.shape}")

            if df_csv.height != embeddings.shape[0]:
                logger.warning(f"{csv_path} ({df_csv.height}) 和 {npy_path} ({embeddings.shape[0]}) 之间的行数不匹配。跳过。")
                continue
                
            # 过滤有效的偏好并映射到标签
            valid_prefs = {'like', 'dislike'}
            df_csv = df_csv.filter(pl.col('preference').is_in(valid_prefs))
            if df_csv.is_empty():
                 logger.warning(f"在{csv_path}中没有找到有效的'like'或'dislike'偏好。跳过。")
                 continue

            df_csv = df_csv.with_columns(
                pl.when(pl.col('preference') == 'like').then(1).otherwise(0).alias('label')
            )
            
            # 添加embedding列
            embedding_lists = [row.tolist() for row in embeddings] 
            df_csv = df_csv.with_columns(pl.Series("embedding", embedding_lists))

            # 过滤掉已加载的ID（保留第一次出现）
            current_ids = set(df_csv['id'].to_list())
            new_ids_mask = df_csv['id'].is_in(loaded_ids).not_()
            df_new = df_csv.filter(new_ids_mask)
            
            if not df_new.is_empty():
                 all_pref_dfs.append(df_new.select(["id", "preference", "label", "embedding"]))
                 newly_added_ids = set(df_new['id'].to_list())
                 loaded_ids.update(newly_added_ids)
                 logger.info(f"从{csv_path}添加了{df_new.height}个新的唯一条目。")
            else:
                 logger.info(f"在{csv_path}中没有找到新的唯一条目。")

        except Exception as e:
            logger.exception(f"处理文件对时出错: {csv_path} / {npy_path}")

    if not all_pref_dfs:
        logger.error("无法加载任何有效的偏好数据。")
        return None

    df_combined_prefs = pl.concat(all_pref_dfs, how="vertical_relaxed") # 使用宽松模式以防模式略有不同
    logger.info(f"合并的偏好数据形状: {df_combined_prefs.shape}")
    return df_combined_prefs

def load_background_data(bg_file_path_str: str) -> Optional[pl.DataFrame]:
    """加载背景数据，允许只提供NPY文件。
    
    Args:
        bg_file_path_str: NPY文件路径或CSV文件路径
    
    Returns:
        包含ID和embedding的DataFrame，如果只有NPY文件则自动生成ID
    """
    # 判断输入是否为NPY文件
    bg_path = pathlib.Path(bg_file_path_str)
    is_npy_input = bg_path.suffix.lower() == '.npy'
    
    if is_npy_input:
        # 直接从NPY加载嵌入向量
        npy_path = bg_path
        csv_path = bg_path.with_suffix(".csv")  # 可能不存在
    else:
        # 传统方式：从CSV加载，并查找对应的NPY
        csv_path = bg_path
        npy_path = bg_path.with_suffix(".npy")
    
    # 检查NPY文件是否存在
    if not npy_path.is_file():
        logger.error(f"背景数据NPY文件不存在: {npy_path}")
        return None
    
    try:
        # 加载嵌入向量
        embeddings = np.load(npy_path)
        logger.info(f"加载背景数据嵌入向量，形状: {embeddings.shape}")
        
        # 如果CSV存在且不是NPY输入模式，尝试加载ID
        if csv_path.is_file() and not is_npy_input:
            try:
                df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
                
                # 检查行数是否匹配
                if df_csv.height != embeddings.shape[0]:
                    logger.warning(f"背景数据行数不匹配: CSV ({df_csv.height}) vs NPY ({embeddings.shape[0]})")
                    logger.warning("将自动生成ID而不使用CSV中的ID")
                    df_csv = None
                elif 'id' not in df_csv.columns:
                    logger.warning(f"背景数据CSV {csv_path} 中缺少'id'列")
                    logger.warning("将自动生成ID而不使用CSV")
                    df_csv = None
            except Exception as e:
                logger.warning(f"加载背景数据CSV时出错: {e}")
                logger.warning("将继续处理，自动生成ID")
                df_csv = None
        else:
            df_csv = None
            if not is_npy_input:
                logger.warning(f"背景数据CSV文件不存在: {csv_path}")
                logger.warning("将继续处理，自动生成ID")
        
        # 如果无法使用CSV数据或者只提供了NPY文件，生成自动ID
        if df_csv is None:
            # 为每个嵌入向量生成唯一ID
            auto_ids = [f"bg_{i:06d}" for i in range(embeddings.shape[0])]
            df_csv = pl.DataFrame({
                "id": auto_ids
            })
            logger.info(f"已为{len(auto_ids)}个背景嵌入向量生成自动ID")
        
        # 将嵌入向量添加到DataFrame
        embedding_lists = [row.tolist() for row in embeddings]
        df_result = df_csv.with_columns(pl.Series("embedding", embedding_lists))
        
        # 如果id列中有空值，进行过滤
        if df_result["id"].is_null().any():
            null_count = df_result["id"].is_null().sum()
            logger.warning(f"背景数据中发现{null_count}个空ID，将被过滤掉")
            df_result = df_result.filter(pl.col("id").is_not_null())
        
        logger.info(f"成功加载背景数据，形状: {df_result.shape}")
        return df_result.select(["id", "embedding"])  # 只保留必要的列
    except Exception as e:
        logger.exception(f"加载背景数据时出错: {e}")
        return None

def load_target_data(target_file_path_str: str) -> Optional[pl.DataFrame]:
    """加载目标数据，允许只提供NPY文件。
    
    Args:
        target_file_path_str: NPY文件路径或CSV文件路径
    
    Returns:
        包含ID和embedding的DataFrame，如果只有NPY文件则自动生成ID
    """
    # 判断输入是否为NPY文件
    target_path = pathlib.Path(target_file_path_str)
    is_npy_input = target_path.suffix.lower() == '.npy'
    
    if is_npy_input:
        # 直接从NPY加载嵌入向量
        npy_path = target_path
        csv_path = target_path.with_suffix(".csv")  # 可能不存在
    else:
        # 传统方式：从CSV加载，并查找对应的NPY
        csv_path = target_path
        npy_path = target_path.with_suffix(".npy")
    
    # 检查NPY文件是否存在
    if not npy_path.is_file():
        logger.error(f"目标数据NPY文件不存在: {npy_path}")
        return None
    
    try:
        # 加载嵌入向量
        embeddings = np.load(npy_path)
        logger.info(f"加载目标数据嵌入向量，形状: {embeddings.shape}")
        
        # 如果CSV存在且不是NPY输入模式，尝试加载完整数据
        if csv_path.is_file() and not is_npy_input:
            try:
                df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
                
                # 检查行数是否匹配
                if df_csv.height != embeddings.shape[0]:
                    logger.warning(f"目标数据行数不匹配: CSV ({df_csv.height}) vs NPY ({embeddings.shape[0]})")
                    logger.warning("将自动生成ID和其他元数据")
                    df_csv = None
                elif 'id' not in df_csv.columns:
                    logger.warning(f"目标数据CSV {csv_path} 中缺少'id'列")
                    logger.warning("将自动生成ID")
                    # 保留其他可能有用的列，只添加id列
                    auto_ids = [f"target_{i:06d}" for i in range(df_csv.height)]
                    df_csv = df_csv.with_columns(pl.Series("id", auto_ids))
            except Exception as e:
                logger.warning(f"加载目标数据CSV时出错: {e}")
                logger.warning("将继续处理，自动生成ID和元数据")
                df_csv = None
        else:
            df_csv = None
            if not is_npy_input:
                logger.warning(f"目标数据CSV文件不存在: {csv_path}")
                logger.warning("将继续处理，自动生成ID和元数据")
        
        # 如果无法使用CSV数据或者只提供了NPY文件，生成自动ID和元数据
        if df_csv is None:
            # 为每个嵌入向量生成唯一ID和基础元数据
            auto_ids = [f"target_{i:06d}" for i in range(embeddings.shape[0])]
            df_csv = pl.DataFrame({
                "id": auto_ids,
                "title": [f"Auto-Generated Title {i}" for i in range(embeddings.shape[0])],
                "abstract": ["No abstract available"] * embeddings.shape[0],
                "date": [datetime.now().isoformat()] * embeddings.shape[0],
                "authors": ["Unknown Author"] * embeddings.shape[0],
                "primary_category": ["unknown"] * embeddings.shape[0]
            })
            logger.info(f"已为{len(auto_ids)}个目标嵌入向量生成自动ID和元数据")
        
        # 将嵌入向量添加到DataFrame
        embedding_lists = [row.tolist() for row in embeddings]
        df_result = df_csv.with_columns(pl.Series("embedding", embedding_lists))
        
        # 如果id列中有空值，进行过滤
        if df_result["id"].is_null().any():
            null_count = df_result["id"].is_null().sum()
            logger.warning(f"目标数据中发现{null_count}个空ID，将被过滤掉")
            df_result = df_result.filter(pl.col("id").is_not_null())
        
        logger.info(f"成功加载目标数据，形状: {df_result.shape}")
        return df_result
    except Exception as e:
        logger.exception(f"加载目标数据时出错: {e}")
        return None


def prepare_training_data(df_prefs: pl.DataFrame, df_bg: pl.DataFrame, neg_ratio: float, random_state: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """准备训练数据，通过从背景中采样负样本。
    
    注意：会自动过滤掉空ID的样本
    """
    
    logger.info("准备训练数据，进行负采样...")
    
    # 过滤掉空ID的样本
    if df_prefs["id"].is_null().any():
        null_count = df_prefs["id"].is_null().sum()
        logger.warning(f"偏好数据中发现{null_count}个空ID，将被过滤掉")
        df_prefs = df_prefs.filter(pl.col("id").is_not_null())
        
        if df_prefs.is_empty():
            logger.error("过滤空ID后，偏好数据为空。无法训练。")
            return None
    
    # 处理空的embedding
    if df_prefs["embedding"].is_null().any():
        null_count = df_prefs["embedding"].is_null().sum()
        logger.warning(f"偏好数据中发现{null_count}个空embedding，将被过滤掉")
        df_prefs = df_prefs.filter(pl.col("embedding").is_not_null())
        
        if df_prefs.is_empty():
            logger.error("过滤空embedding后，偏好数据为空。无法训练。")
            return None
    
    # 分离正样本和负样本
    df_pos = df_prefs.filter(pl.col('label') == 1)
    df_neg_explicit = df_prefs.filter(pl.col('label') == 0)
    
    n_positive = df_pos.height
    n_neg_explicit = df_neg_explicit.height
    
    if n_positive == 0:
        logger.error("没有找到正样本。无法训练。")
        return None

    # 从背景池中移除偏好样本ID
    pref_ids = set(df_prefs['id'].to_list())
    df_bg_pool = df_bg.filter(pl.col('id').is_in(pref_ids).not_())
    n_available_bg = df_bg_pool.height
    
    if n_available_bg == 0:
        logger.warning("移除偏好ID后，没有可用的背景样本。")
        n_to_sample = 0
    else:
        # 直接采样neg_ratio倍的正样本数量
        n_to_sample = int(n_positive * neg_ratio)
        n_to_sample = min(n_to_sample, n_available_bg)  # 不能超过可用的背景样本数
        logger.info(f"采样逻辑 - 正样本: {n_positive}, 显式负样本: {n_neg_explicit}, 采样负样本: {n_to_sample} (从 {n_available_bg} 背景样本中)")

    # 采样负样本
    if n_to_sample > 0:
        df_neg_sampled = df_bg_pool.sample(n=n_to_sample, shuffle=True, seed=random_state)
        df_neg_sampled = df_neg_sampled.with_columns(pl.lit(0).alias('label'))
        df_neg_sampled = df_neg_sampled.select(["id", "embedding", "label"]) 
    else:
        df_neg_sampled = pl.DataFrame({"id": [], "embedding": [], "label": []}, schema={"id": pl.Utf8, "embedding": pl.List(pl.Float32), "label": pl.Int8})

    # 合并数据
    df_train = pl.concat([
        df_pos.select(["id", "embedding", "label"]), 
        df_neg_explicit.select(["id", "embedding", "label"]), 
        df_neg_sampled
    ], how="vertical_relaxed")

    # 最终检查和转换为NumPy数组
    if df_train.is_empty():
        logger.error("处理后的训练数据为空。")
        return None
        
    final_counts = df_train['label'].value_counts()
    pos_count = final_counts.filter(pl.col('label') == 1).select('count').item(0,0) if 1 in final_counts['label'] else 0
    neg_count = final_counts.filter(pl.col('label') == 0).select('count').item(0,0) if 0 in final_counts['label'] else 0
    
    if pos_count == 0 or neg_count == 0:
         logger.error(f"训练数据缺少两个类别的样本 (正: {pos_count}, 负: {neg_count})。无法训练。")
         return None
         
    logger.info(f"最终训练数据分布 - 正样本: {pos_count}, 负样本(显式+采样): {neg_count}, 比例: {neg_count/pos_count:.2f}:1")
    
    try:
        # 确保embedding不为空
        df_train = df_train.filter(pl.col('embedding').is_not_null())
        if df_train.is_empty():
             logger.error("过滤空embedding后，训练数据为空。")
             return None
             
        X = np.array(df_train['embedding'].to_list(), dtype=np.float64)
        y = df_train['label'].to_numpy()

        # 检查是否有NaN/Inf
        if np.isnan(X).any():
            nan_rows = np.isnan(X).any(axis=1)
            logger.error(f"训练特征中发现NaN值，共{np.sum(nan_rows)}行")
            return None
        if np.isinf(X).any():
            inf_rows = np.isinf(X).any(axis=1)
            logger.error(f"训练特征中发现无穷值，共{np.sum(inf_rows)}行")
            return None

        logger.info(f"准备好的训练特征形状: X: {X.shape}, y: {y.shape}")
        return X, y
    except Exception as e:
        logger.exception(f"转换训练数据为NumPy数组时出错: {e}")
        return None


def perform_cv_and_get_threshold(X: np.ndarray, y: np.ndarray, n_splits: int, beta: float, random_state: int, C: float = 1.0, max_iter: int = 1000) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Performs K-Fold CV, logs metrics, returns overall optimal threshold and cv probabilities."""
    logger.info(f"Starting {n_splits}-fold cross-validation (beta={beta}, C={C}, max_iter={max_iter})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_true = []
    all_probs = []
    fold_aucs = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 使用逻辑回归模型
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, 
                                  class_weight='balanced')
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
        logger.info(f"训练集正样本比例: {np.mean(y_train):.3f}, 测试集正样本比例: {np.mean(y_test):.3f}") # loguru 输出

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

    return overall_threshold, np.array(all_true), np.array(all_probs)


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


def dynamic_margin_biased_sample(scores: np.ndarray, threshold: float, y_true=None, y_prob=None, min_margin: float = 0.05, max_margin: float = 0.2) -> np.ndarray:
    """Performs biased sampling with dynamic margin based on AUC curve characteristics."""
    
    # 1. 计算动态margin
    if y_true is not None and y_prob is not None:
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # 找到最接近optimal_threshold的索引
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        logger.info(f"找到最接近阈值({threshold:.4f})的点，索引: {threshold_idx}/{len(thresholds)-1}")
        
        # 计算曲线特征 - 改进的曲率计算方法
        try:
            # 1. 使用窗口平均斜率，而不是单点斜率
            window_size = 3  # 窗口大小，保证平滑性
            slope_values = []
            vicinity_indices = []
            
            # 收集阈值附近的点
            start_idx = max(0, threshold_idx - window_size)
            end_idx = min(len(fpr) - 1, threshold_idx + window_size)
            vicinity_indices = list(range(start_idx, end_idx+1))
            
            # 计算窗口内所有点的斜率
            for i in range(start_idx, end_idx):
                if fpr[i+1] - fpr[i] < 1e-6:  # 分母保护
                    continue
                slope = (tpr[i+1] - tpr[i]) / (fpr[i+1] - fpr[i])
                if abs(slope) < 1e6:  # 过滤掉极端斜率值
                    slope_values.append(slope)
            
            if not slope_values:
                logger.warning("无法计算有效斜率，使用默认margin")
                vicinity_margin = (min_margin + max_margin) / 2
            else:
                # 2. 曲线形状分析 - 基于斜率变化率的凹凸性
                max_slope = max(slope_values) if slope_values else 0
                min_slope = min(slope_values) if slope_values else 0
                slope_range = max_slope - min_slope
                
                # 3. 曲线决策边界清晰度 - 斜率范围反映了边界清晰度
                # 范围小 = 平坦曲线 = 需要大margin来增加多样性
                # 范围大 = 陡峭曲线 = 用小margin保持高质量
                
                # 归一化斜率范围 - 更稳健的方式
                norm_factor = 10.0  # 控制归一化程度
                curve_clarity = np.tanh(slope_range / norm_factor)  # tanh确保[0,1]范围，避免极端值
                
                # 根据曲线特征确定margin
                # 曲线清晰度低(平坦) → 大margin，曲线清晰度高(陡峭) → 小margin
                vicinity_margin = max_margin - curve_clarity * (max_margin - min_margin)
                
                # 记录详细信息
                logger.info(f"曲线分析 - 窗口大小: {window_size}, 有效斜率个数: {len(slope_values)}")
                logger.info(f"曲线特征 - 最大斜率: {max_slope:.4f}, 最小斜率: {min_slope:.4f}, 斜率范围: {slope_range:.4f}")
                logger.info(f"归一化曲线清晰度: {curve_clarity:.4f}, 映射margin: {vicinity_margin:.4f}")
        except Exception as e:
            logger.warning(f"计算曲率时出错: {e}，使用默认margin")
            vicinity_margin = (min_margin + max_margin) / 2
    else:
        # 无AUC信息时使用默认边界
        vicinity_margin = (min_margin + max_margin) / 2
        logger.info(f"无AUC曲线信息，使用默认margin: {vicinity_margin:.4f}")
    
    logger.info(f"动态计算vicinity_margin = {vicinity_margin:.4f}")
    
    # 2. 调用原始采样函数执行采样
    return biased_sample(scores, threshold, vicinity_margin)


def display_sample_papers(df: pl.DataFrame, threshold: float, n_samples: int = 5) -> None:
    """显示不同分数范围的样本论文。

    Args:
        df: 包含论文信息和预测分数的DataFrame
        threshold: 模型预测的阈值
        n_samples: 每个类别显示的样本数
    """
    if n_samples <= 0 or df.is_empty():
        return

    # 确保必要的列存在
    required_cols = ['id', 'score']
    optional_cols = ['title', 'abstract', 'authors', 'date', 'primary_category']
    
    if not all(col in df.columns for col in required_cols):
        logger.error(f"缺少必要的列，无法显示样本。需要: {required_cols}")
        return
    
    available_info_cols = [col for col in optional_cols if col in df.columns]
    
    # 定义分数范围
    high_scores = df.filter(pl.col('score') >= threshold + 0.2)
    medium_scores = df.filter((pl.col('score') >= threshold - 0.1) & 
                            (pl.col('score') < threshold + 0.1))
    low_scores = df.filter(pl.col('score') < threshold - 0.2)
    
    # 抽样展示
    for category, df_category, name in [
        ("高分论文", high_scores, "HIGH"), 
        ("中分论文", medium_scores, "MEDIUM"), 
        ("低分论文", low_scores, "LOW")
    ]:
        if not df_category.is_empty():
            samples = df_category.sample(n=min(n_samples, df_category.height), shuffle=True, seed=42)
            logger.info(f"\n{'='*50}\n{name} SCORE PAPERS\n{'='*50}")
            
            for row in samples.sort(pl.col('score'), descending=True).rows(named=True):
                logger.info(f"\n论文ID: {row['id']} - 分数: {row['score']:.4f}")
                
                if 'title' in row and row['title']:
                    logger.info(f"标题: {row['title']}")
                
                if 'abstract' in row and row['abstract']:
                    abstract = row['abstract']
                    # 截断过长的摘要
                    if len(abstract) > 500:
                        abstract = abstract[:500] + "..."
                    logger.info(f"摘要: {abstract}")
                
                if 'authors' in row and row['authors']:
                    logger.info(f"作者: {row['authors']}")
                
                if 'date' in row and row['date']:
                    logger.info(f"发布日期: {row['date']}")
                
                if 'primary_category' in row and row['primary_category']:
                    logger.info(f"主要类别: {row['primary_category']}")
                
                # 添加分隔线
                logger.info("-" * 30)


# --- Main Script Logic ---
def main():
    # 配置中文字体支持
    configure_chinese_font()
    
    # Load the entire config
    full_config = load_config()
    # Get specific sections, defaulting to empty dict if section is missing
    model_fitting_cfg = full_config.get('model_fitting', {})

    parser = argparse.ArgumentParser(description="训练偏好模型，预测分数，并进行采样。从config.toml [model_fitting]读取默认值。")
    # Updated arguments to read defaults from model_fitting_cfg
    parser.add_argument("--preference-dir", "-p", type=str, 
                        default=model_fitting_cfg.get('preference_dir'),
                        help="包含偏好CSV+NPY文件的目录。")
    parser.add_argument("--background-file", "-b", type=str, 
                        default=model_fitting_cfg.get('background_file'),
                        help="背景数据文件路径，可以是CSV文件或NPY文件。如果提供NPY文件，会自动生成ID。")
    parser.add_argument("--target-file", "-t", type=str, 
                        default=model_fitting_cfg.get('target_file'),
                        help="目标预测文件路径，可以是CSV文件或NPY文件。如果提供NPY文件，会自动生成ID和元数据。")
    parser.add_argument("--neg-ratio", type=float, 
                        default=model_fitting_cfg.get('neg_ratio', 5.0),
                        help="Target ratio of negative to positive samples after sampling.")
    parser.add_argument("--folds", "-k", type=int, 
                        default=model_fitting_cfg.get('folds', 5),
                        help="Number of folds for cross-validation.")
    parser.add_argument("--beta", type=float, 
                        default=model_fitting_cfg.get('beta', 1.2),
                        help="Beta value for F-beta/G-mean optimization in threshold finding.")
    # 添加手动设置vicinity_margin的参数
    parser.add_argument("--vicinity-margin", type=float, 
                        default=model_fitting_cfg.get('vicinity_margin', 0.2),
                        help="Manually set the vicinity margin for biased sampling (overrides dynamic calculation).")
    parser.add_argument("--random-state", type=int, 
                        default=model_fitting_cfg.get('random_state', 42),
                        help="Random state for reproducibility.")
    # 添加展示样本的参数
    parser.add_argument("--sample", type=int,
                        default=model_fitting_cfg.get('sample', 0),
                        help="Number of sample papers to display from each score category (0 to disable).")
    # 添加绘图参数
    parser.add_argument("--plots-dir", type=str,
                        default=None,
                        help="Directory to save visualization plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable generating plots")

    args = parser.parse_args()

    # --- Set Random Seeds for Reproducibility ---
    seed = args.random_state
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seeds (python, numpy) to: {seed}")

    # --- 创建绘图目录（如果启用） ---
    plots_enabled = not args.no_plots
    plots_dir = None
    if plots_enabled and args.plots_dir:
        plots_dir = pathlib.Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"将保存可视化图表到: {plots_dir}")

    # --- Load Data --- 
    logger.info("--- Loading Data ---")
    # Check required arguments after parsing (considering defaults from config)
    pref_dir_str = args.preference_dir
    bg_file_str = args.background_file # Renamed for clarity
    target_file_str = args.target_file   # Renamed for clarity

    missing_required = []
    if not pref_dir_str:
        missing_required.append('--preference-dir/-p')
    if not bg_file_str:
        missing_required.append('--background-file/-b')
    if not target_file_str:
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
    df_bg = load_background_data(bg_file_str)
    if df_bg is None:
        logger.error("Failed to load background data. Cannot proceed with negative sampling.")
        sys.exit(1)

    # Pass the full CSV path string to the loading function
    df_target = load_target_data(target_file_str)
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
    cv_C = 1
    cv_max_iter = 10000
    cv_result = perform_cv_and_get_threshold(
        X_train, y_train, args.folds, args.beta, args.random_state, C=cv_C, max_iter=cv_max_iter
    )
    if cv_result is None:
        logger.error("Failed to determine optimal threshold from CV. Exiting.")
        sys.exit(1)
    
    optimal_threshold, cv_true, cv_probs = cv_result
    
    # 使用固定阈值替代自动计算的阈值
    fixed_threshold = 0.5
    logger.info(f"自动计算的最佳阈值为: {optimal_threshold:.4f}，但将使用固定阈值: {fixed_threshold}")

    # --- Final Model Training ---
    logger.info("--- Training Final Model --- ")
    # 使用逻辑回归模型
    final_model = LogisticRegression(C=cv_C, max_iter=cv_max_iter, random_state=args.random_state, 
                                    class_weight='balanced')
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

    # --- 在背景数据上预测分数 ---
    logger.info("--- 在背景数据上预测分数 ---")
    try:
        X_bg = np.array(df_bg['embedding'].to_list(), dtype=np.float32)
        
        if np.isnan(X_bg).any() or np.isinf(X_bg).any():
            logger.warning("背景数据中存在NaN或Inf值，可能影响预测质量")
        
        bg_scores = final_model.predict_proba(X_bg)[:, 1]
        df_bg = df_bg.with_columns(pl.Series("score", bg_scores))
        logger.info(f"成功为{len(bg_scores)}个背景样本计算预测分数")
    except Exception as e:
        logger.exception("在背景数据上预测分数时出错")
        bg_scores = np.array([])

    # --- 获取训练数据的分数 ---
    logger.info("--- 获取训练数据的分数 ---")
    try:
        train_scores = final_model.predict_proba(X_train)[:, 1]
        logger.info(f"成功为{len(train_scores)}个训练样本获取预测分数")
        
        # 分离正样本和负样本分数
        pos_scores = train_scores[y_train == 1]
        neg_scores = train_scores[y_train == 0]
        logger.info(f"训练集中正样本:{len(pos_scores)}个，负样本:{len(neg_scores)}个")
    except Exception as e:
        logger.exception("获取训练数据分数时出错")
        train_scores = np.array([])
        pos_scores = np.array([])
        neg_scores = np.array([])

    # --- 绘制分布图表 ---
    if plots_enabled:
        logger.info("--- 生成可视化图表 ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 绘制分数分布图
        try:
            scores_dict = {
                "目标数据": target_scores,
                "背景数据": bg_scores,
                "训练数据(正样本)": pos_scores
            }
            
            scores_path = None
            if plots_dir:
                scores_path = plots_dir / f"score_distributions_{timestamp}.png"
            
            plot_score_distributions(scores_dict, fixed_threshold, scores_path)
            logger.info("成功绘制分数分布图")
        except Exception as e:
            logger.exception("绘制分数分布图时出错")
        
        # 绘制ROC曲线
        try:
            roc_path = None
            if plots_dir:
                roc_path = plots_dir / f"roc_curve_{timestamp}.png"
                
            plot_roc_curve(cv_true, cv_probs, fixed_threshold, args.beta, roc_path)
            logger.info("成功绘制ROC曲线")
        except Exception as e:
            logger.exception("绘制ROC曲线时出错")

    # --- Biased Sampling ---
    logger.info("--- Performing Biased Sampling ---")
    
    # 使用固定阈值进行采样
    logger.info(f"使用固定阈值 {fixed_threshold} 和阈值边界 {args.vicinity_margin} 进行采样")
    show_flags = biased_sample(
        target_scores, 
        fixed_threshold,
        vicinity_margin=args.vicinity_margin
    )
    
    df_target = df_target.with_columns(pl.Series("show", show_flags))

    # --- Save Output ---
    logger.info("--- Saving Results ---")
    # Determine the output path - always overwrite the target CSV
    # Use the input target CSV path string directly
    output_pred_path = pathlib.Path(target_file_str)
    logger.warning(f"Output CSV path set to overwrite target file: {output_pred_path}")
        
    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Drop embedding column before saving
        df_target_to_save = df_target.drop("embedding") 
        df_target_to_save.write_csv(output_pred_path)
        logger.info(f"Predictions and sampling results saved, overwriting: {output_pred_path}")
    except Exception as e:
        logger.exception(f"Failed to save prediction results to {output_pred_path}")

    # --- 显示样本论文 ---
    if args.sample > 0:
        logger.info(f"--- 显示样本论文 (每类 {args.sample} 篇) ---")
        display_sample_papers(df_target, fixed_threshold, n_samples=args.sample)

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
#   --plots-dir data/visualizations

""" 