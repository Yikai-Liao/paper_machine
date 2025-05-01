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
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

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

# --- 新增边界优化采样策略实现 ---
def borderline_smote(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5, sampling_ratio: float = 1.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Implements Borderline-SMOTE algorithm for boundary-optimized oversampling.
    
    Args:
        X: Feature matrix, shape is (n_samples, n_features)
        y: Label vector, shape is (n_samples,)
        n_neighbors: Number of neighbors to use for determining boundary samples
        sampling_ratio: Oversampling ratio, number of synthetic samples = original positive sample count * this ratio
        random_state: Random seed
        
    Returns:
        Oversampled feature matrix and label vector
    """
    logger.info(f"Executing Borderline-SMOTE boundary optimization oversampling, parameters: n_neighbors={n_neighbors}, sampling_ratio={sampling_ratio}")
    np.random.seed(random_state)
    
    # Separate positive and negative samples
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    n_pos = X_pos.shape[0]
    n_neg = X_neg.shape[0]
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("Cannot execute Borderline-SMOTE: Missing positive or negative samples in data")
        return X, y
        
    logger.info(f"Original data: {n_pos} positive samples, {n_neg} negative samples")
    
    # Find k nearest neighbors for positive samples
    if n_neighbors > n_pos:
        logger.warning(f"n_neighbors ({n_neighbors}) is greater than the number of positive samples ({n_pos}), adjusting to {n_pos-1}")
        n_neighbors = max(1, n_pos - 1)
        
    # Calculate neighbors for each positive sample
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)  # +1 because the sample itself is counted
    nn.fit(np.vstack((X_pos, X_neg)))
    
    # Find k nearest neighbors for each positive sample
    distances, indices = nn.kneighbors(X_pos)
    
    # Identify boundary samples: more than half of the k neighbors of a positive sample are negative samples
    border_samples_idx = []
    for i in range(n_pos):
        # Skip the first neighbor (the sample itself)
        neighbor_indices = indices[i, 1:]
        # Count negative neighbors
        n_neg_neighbors = sum(1 for idx in neighbor_indices if idx >= n_pos)
        danger_ratio = n_neg_neighbors / n_neighbors
        
        # Boundary check: if more than half of the neighbors are negative
        if 0 < danger_ratio < 1.0:  # Not all negative neighbors, at least some positive neighbors
            border_samples_idx.append(i)
    
    n_border = len(border_samples_idx)
    logger.info(f"Identified {n_border} boundary positive samples (proportion: {n_border/n_pos*100:.1f}%)")
    
    if n_border == 0:
        logger.warning("No boundary samples found, cannot perform oversampling")
        return X, y
    
    # Determine the number of synthetic samples to generate
    n_samples_to_generate = int(n_border * sampling_ratio)
    if n_samples_to_generate <= 0:
        logger.warning(f"Calculated number of synthetic samples is {n_samples_to_generate}, not performing oversampling")
        return X, y
        
    logger.info(f"Preparing to generate {n_samples_to_generate} synthetic boundary samples")
    
    # Given boundary samples, find k positive neighbors for each boundary sample
    border_X = X_pos[border_samples_idx]
    
    # Find neighbors only among positive samples
    nn_pos = NearestNeighbors(n_neighbors=min(n_neighbors+1, n_pos))
    nn_pos.fit(X_pos)
    distances_pos, indices_pos = nn_pos.kneighbors(border_X)
    
    # Start generating synthetic samples
    synthetic_X = []
    
    # Generate synthetic samples for boundary samples according to the sampling ratio
    for _ in range(n_samples_to_generate):
        # Randomly select a boundary sample
        idx = np.random.randint(0, n_border)
        
        # Randomly select one of its positive neighbors (skip the first one, which is itself)
        nn_idx = np.random.choice(indices_pos[idx, 1:])
        
        # Generate synthetic sample
        diff = X_pos[nn_idx] - border_X[idx]
        gap = np.random.random()
        synthetic_sample = border_X[idx] + gap * diff
        
        synthetic_X.append(synthetic_sample)
    
    if len(synthetic_X) > 0:
        synthetic_X = np.array(synthetic_X)
        X_resampled = np.vstack((X, synthetic_X))
        y_resampled = np.hstack((y, np.ones(len(synthetic_X))))
        
        logger.info(f"Data after oversampling: Total samples {X_resampled.shape[0]}, added synthetic samples {len(synthetic_X)}")
        return X_resampled, y_resampled
    else:
        logger.warning("Failed to generate any synthetic samples")
        return X, y

def adasyn(X: np.ndarray, y: np.ndarray, beta: float = 1.0, n_neighbors: int = 5, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Implements ADASYN (Adaptive Synthetic Sampling) algorithm.
    
    Args:
        X: Feature matrix, shape is (n_samples, n_features)
        y: Label vector, shape is (n_samples,)
        beta: Proportion of synthetic samples to generate relative to class imbalance
        n_neighbors: Number of neighbors to use for density estimation
        random_state: Random seed
        
    Returns:
        Oversampled feature matrix and label vector
    """
    logger.info(f"Executing ADASYN adaptive synthetic sampling, parameters: beta={beta}, n_neighbors={n_neighbors}")
    np.random.seed(random_state)
    
    # Separate positive and negative samples
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    n_pos = X_pos.shape[0]
    n_neg = X_neg.shape[0]
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("Cannot execute ADASYN: Missing positive or negative samples in data")
        return X, y
        
    logger.info(f"Original data: {n_pos} positive samples, {n_neg} negative samples")
    
    # Calculate class imbalance ratio
    if n_pos > n_neg:
        logger.info("Number of positive samples is greater than negative samples, no oversampling needed")
        return X, y
    
    imbalance_ratio = n_neg / n_pos
    # Calculate the total number of synthetic samples needed
    G = int((n_neg - n_pos) * beta)
    if G <= 0:
        logger.warning(f"Calculated number of synthetic samples is {G}, not performing oversampling")
        return X, y
    
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}, planned number of synthetic samples to generate: {G}")
    
    # Calculate difficulty level (r_i) for each positive sample, i.e., the proportion of negative samples among its k nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)  # +1 because the sample itself is counted
    nn.fit(np.vstack((X_pos, X_neg)))
    
    distances, indices = nn.kneighbors(X_pos)
    
    r_i = []
    for i in range(n_pos):
        # Skip the first neighbor (the sample itself)
        neighbor_indices = indices[i, 1:]
        # Count negative neighbors
        n_neg_neighbors = sum(1 for idx in neighbor_indices if idx >= n_pos)
        r_i.append(n_neg_neighbors / n_neighbors)
    
    r_i = np.array(r_i)
    
    # Normalize r_i so that they sum to 1
    if np.sum(r_i) == 0:
        logger.warning("Difficulty level for all positive samples is 0, cannot perform adaptive oversampling")
        return X, y
    
    r_i = r_i / np.sum(r_i)
    
    # Calculate the number of synthetic samples to generate for each positive sample
    n_samples_per_pos = np.round(r_i * G).astype(int)
    
    # Find k nearest neighbors for positive samples (only among positive samples)
    nn_pos = NearestNeighbors(n_neighbors=min(n_neighbors, n_pos))
    nn_pos.fit(X_pos)
    
    synthetic_X = []
    
    # Generate the corresponding number of synthetic samples for each positive sample
    for i, n_samples in enumerate(n_samples_per_pos):
        if n_samples == 0:
            continue
            
        # Find the k nearest neighbors of the positive sample (all are positive samples)
        distances_pos, indices_pos = nn_pos.kneighbors(X_pos[i].reshape(1, -1))
        
        # Generate n_samples synthetic samples
        for _ in range(n_samples):
            # Randomly select one of the neighbors (skip the first one, which is itself)
            if len(indices_pos[0]) <= 1:  # Only itself, no other neighbors
                continue
            
            nn_idx = np.random.choice(indices_pos[0][1:])
            
            # Generate synthetic sample
            diff = X_pos[nn_idx] - X_pos[i]
            gap = np.random.random()
            synthetic_sample = X_pos[i] + gap * diff
            
            synthetic_X.append(synthetic_sample)
    
    if len(synthetic_X) > 0:
        synthetic_X = np.array(synthetic_X)
        X_resampled = np.vstack((X, synthetic_X))
        y_resampled = np.hstack((y, np.ones(len(synthetic_X))))
        
        logger.info(f"Data after oversampling: Total samples {X_resampled.shape[0]}, added synthetic samples {len(synthetic_X)}")
        return X_resampled, y_resampled
    else:
        logger.warning("Failed to generate any synthetic samples")
        return X, y

def confidence_weighted_sampling(df: pl.DataFrame, model, high_conf_threshold: float = 0.9, high_conf_weight: float = 2.0, n_samples: Optional[int] = None, random_state: int = 42) -> pl.DataFrame:
    """Perform confidence-weighted sampling on positive samples.
    
    Args:
        df: DataFrame containing sample information, must include 'embedding' and 'label' columns
        model: Trained model used to calculate sample confidence
        high_conf_threshold: High confidence threshold, samples above this are considered high confidence
        high_conf_weight: Weight multiplier for high confidence samples
        n_samples: Number of samples to sample, if None, sample all samples
        random_state: Random seed
    
    Returns:
        Sampled DataFrame
    """
    if df.is_empty():
        logger.warning("Input DataFrame is empty, cannot perform confidence-weighted sampling")
        return df
    
    # Only perform confidence-weighted sampling on positive samples
    df_pos = df.filter(pl.col('label') == 1)
    if df_pos.is_empty():
        logger.warning("No positive samples, cannot perform confidence-weighted sampling")
        return df
    
    # If the number of samples is not specified, default to sampling all samples
    if n_samples is None:
        n_samples = df_pos.height
    else:
        n_samples = min(n_samples, df_pos.height)
    
    # Convert embeddings to numpy array and predict confidence
    X_pos = np.array(df_pos['embedding'].to_list())
    try:
        pos_confidences = model.predict_proba(X_pos)[:, 1]
        logger.info(f"Calculated confidence scores for {len(pos_confidences)} positive samples")
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        # If confidence cannot be calculated, use uniform weights
        return df_pos.sample(n=n_samples, shuffle=True, seed=random_state)
    
    # Calculate sampling weights based on confidence
    weights = np.ones_like(pos_confidences)
    high_conf_indices = np.where(pos_confidences >= high_conf_threshold)[0]
    weights[high_conf_indices] = high_conf_weight
    
    # Count high confidence samples
    n_high_conf = len(high_conf_indices)
    logger.info(f"Found {n_high_conf} high confidence positive samples (confidence >= {high_conf_threshold}). " + 
                f"Weight setting: High confidence samples={high_conf_weight}, others=1.0")
    
    # Perform weighted random sampling based on weights
    sampling_probs = weights / weights.sum()
    
    # Use numpy's choice function for weighted random sampling
    sampled_indices = np.random.RandomState(random_state).choice(
        range(df_pos.height), size=n_samples, replace=True, p=sampling_probs
    )
    
    # 计算每个高置信度样本被采样的次数，用于日志记录
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    high_conf_sampled = sum(counts[i] for i, idx in enumerate(unique_indices) if idx in high_conf_indices)
    
    logger.info(f"完成置信度加权采样: 总样本量={n_samples}, 其中高置信度样本={high_conf_sampled} ({high_conf_sampled/n_samples*100:.1f}%)")
    
    # 返回采样结果
    return df_pos.sample(n=n_samples, shuffle=True, seed=random_state, with_replacement=True, weights=sampling_probs)

# --- 配置matplotlib中文字体支持 ---
# def configure_chinese_font():
#     """配置matplotlib支持中文字体"""
#     if platform.system() == "Windows":
#         # Windows系统优先使用微软雅黑
#         chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun']
#     elif platform.system() == "Darwin":
#         # macOS系统
#         chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
#     else:
#         # Linux或其他系统
#         chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC']
#     
#     # 尝试设置中文字体
#     for font in chinese_fonts:
#         try:
#             plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
#             plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#             logger.info(f"设置matplotlib中文字体: {font}")
#             # 测试字体是否成功设置
#             fig, ax = plt.subplots(figsize=(1, 1))
#             ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
#             plt.close(fig)  # 关闭测试图形
#             return True
#         except Exception as e:
#             logger.debug(f"尝试设置字体{font}失败: {e}")
#     
#     # 如果所有字体都失败，使用特殊方法
#     try:
#         logger.warning("尝试使用matplotlib内部字体管理器解决中文显示问题")
#         plt.rcParams['font.family'] = 'sans-serif'
#         plt.rcParams['axes.unicode_minus'] = False
#         return True
#     except Exception as e:
#         logger.error(f"配置中文字体失败: {e}")
#         return False

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
    keys_to_keep = ["Target Data", "Background Data", "Training Data (Positive Samples)"]
    
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
               linewidth=2, label='Threshold = 0.5')
    
    # 设置x轴范围为[0,1]
    plt.xlim(0, 1)
    plt.title('Score Distributions', fontsize=15)
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存或显示图表
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Score distribution plot saved to: {output_path}")
        plt.close()
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
    plt.plot(fpr, tpr, color='blue', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    
    # 增大标记点和改善标签可见性
    if threshold_idx < len(fpr):
        plt.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red', s=120, zorder=10,
                    label=f'Threshold = {threshold:.3f}\nFPR = {fpr[threshold_idx]:.3f}\nTPR = {tpr[threshold_idx]:.3f}')
    
    # 设置坐标轴范围和标签
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve Analysis (AUC = {roc_auc:.3f})', fontsize=15)
    
    # 添加网格线以提高可读性
    plt.grid(True, alpha=0.4, linestyle=':')
    
    # 优化图例位置和样式
    plt.legend(loc="lower right", fontsize=11, framealpha=0.8)
    
    # 保存或显示图表
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve plot saved to: {output_path}")
        plt.close()
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
    logger.info(f"Found {len(csv_files)} potential CSV files in {pref_dir}.")

    if not csv_files:
        logger.error(f"No CSV files found in preference directory: {pref_dir}")
        return None

    loaded_ids = set()  # 跟踪已加载的ID，避免跨文件重复

    for csv_path in csv_files:
        npy_path = csv_path.with_suffix('.npy')
        logger.debug(f"Processing CSV: {csv_path}")
        if not npy_path.is_file():
            logger.warning(f"No matching NPY file found: {csv_path}. Skipping this file.")
            continue

        try:
            df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
            logger.debug(f"Loaded CSV {csv_path}, shape {df_csv.shape}")
            
            # 过滤空ID
            if df_csv["id"].is_null().any():
                null_count = df_csv["id"].is_null().sum()
                logger.warning(f"Found {null_count} null IDs in {csv_path}, which will be filtered out")
                df_csv = df_csv.filter(pl.col("id").is_not_null())
                
                if df_csv.is_empty():
                    logger.warning(f"After filtering null IDs, no remaining data in {csv_path}. Skipping this file.")
                    continue
                    
            required_cols = {'id', 'preference'}
            if not required_cols.issubset(df_csv.columns):
                logger.warning(f"CSV {csv_path} is missing required columns ({required_cols}). Skipping.")
                continue

            embeddings = np.load(npy_path)
            logger.debug(f"Loaded NPY {npy_path}, shape {embeddings.shape}")

            if df_csv.height != embeddings.shape[0]:
                logger.warning(f"Mismatch in row count between {csv_path} ({df_csv.height}) and {npy_path} ({embeddings.shape[0]}). Skipping.")
                continue
                
            # 过滤有效的偏好并映射到标签
            valid_prefs = {'like', 'dislike'}
            df_csv = df_csv.filter(pl.col('preference').is_in(valid_prefs))
            if df_csv.is_empty():
                 logger.warning(f"No valid 'like' or 'dislike' preferences found in {csv_path}. Skipping.")
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
                 logger.info(f"Added {df_new.height} new unique entries from {csv_path}.")
            else:
                 logger.info(f"No new unique entries found in {csv_path}.")

        except Exception as e:
            logger.exception(f"Error processing files: {csv_path} / {npy_path}")

    if not all_pref_dfs:
        logger.error("Unable to load any valid preference data.")
        return None

    df_combined_prefs = pl.concat(all_pref_dfs, how="vertical_relaxed") # 使用宽松模式以防模式略有不同
    logger.info(f"Combined preference data shape: {df_combined_prefs.shape}")
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
        logger.error(f"Background data NPY file does not exist: {npy_path}")
        return None
    
    try:
        # 加载嵌入向量
        embeddings = np.load(npy_path)
        logger.info(f"Loaded background data embeddings, shape: {embeddings.shape}")
        
        # 如果CSV存在且不是NPY输入模式，尝试加载ID
        if csv_path.is_file() and not is_npy_input:
            try:
                df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
                
                # 检查行数是否匹配
                if df_csv.height != embeddings.shape[0]:
                    logger.warning(f"Mismatch in row count between CSV ({df_csv.height}) and NPY ({embeddings.shape[0]})")
                    logger.warning("Will generate IDs instead of using CSV IDs")
                    df_csv = None
                elif 'id' not in df_csv.columns:
                    logger.warning(f"'id' column missing in background data CSV {csv_path}")
                    logger.warning("Will generate IDs instead of using CSV")
                    df_csv = None
            except Exception as e:
                logger.warning(f"Error loading background data CSV: {e}")
                logger.warning("Will continue processing, generating IDs")
                df_csv = None
        else:
            df_csv = None
            if not is_npy_input:
                logger.warning(f"Background data CSV file does not exist: {csv_path}")
                logger.warning("Will continue processing, generating IDs")
        
        # 如果无法使用CSV数据或者只提供了NPY文件，生成自动ID
        if df_csv is None:
            # 为每个嵌入向量生成唯一ID
            auto_ids = [f"bg_{i:06d}" for i in range(embeddings.shape[0])]
            df_csv = pl.DataFrame({
                "id": auto_ids
            })
            logger.info(f"Generated automatic IDs for {len(auto_ids)} background embeddings")
        
        # 将嵌入向量添加到DataFrame
        embedding_lists = [row.tolist() for row in embeddings]
        df_result = df_csv.with_columns(pl.Series("embedding", embedding_lists))
        
        # 如果id列中有空值，进行过滤
        if df_result["id"].is_null().any():
            null_count = df_result["id"].is_null().sum()
            logger.warning(f"Found {null_count} null IDs in background data, which will be filtered out")
            df_result = df_result.filter(pl.col("id").is_not_null())
        
        logger.info(f"Successfully loaded background data, shape: {df_result.shape}")
        return df_result.select(["id", "embedding"])  # 只保留必要的列
    except Exception as e:
        logger.exception(f"Error loading background data: {e}")
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
        logger.error(f"Target data NPY file does not exist: {npy_path}")
        return None
    
    try:
        # 加载嵌入向量
        embeddings = np.load(npy_path)
        logger.info(f"Loaded target data embeddings, shape: {embeddings.shape}")
        
        # 如果CSV存在且不是NPY输入模式，尝试加载完整数据
        if csv_path.is_file() and not is_npy_input:
            try:
                df_csv = pl.read_csv(csv_path, schema_overrides={'id': pl.Utf8})
                
                # 检查行数是否匹配
                if df_csv.height != embeddings.shape[0]:
                    logger.warning(f"Mismatch in row count between CSV ({df_csv.height}) and NPY ({embeddings.shape[0]})")
                    logger.warning("Will generate IDs and metadata instead of using CSV")
                    df_csv = None
                elif 'id' not in df_csv.columns:
                    logger.warning(f"'id' column missing in target data CSV {csv_path}")
                    logger.warning("Will generate IDs instead of using CSV")
                    # 保留其他可能有用的列，只添加id列
                    auto_ids = [f"target_{i:06d}" for i in range(df_csv.height)]
                    df_csv = df_csv.with_columns(pl.Series("id", auto_ids))
            except Exception as e:
                logger.warning(f"Error loading target data CSV: {e}")
                logger.warning("Will continue processing, generating IDs and metadata")
                df_csv = None
        else:
            df_csv = None
            if not is_npy_input:
                logger.warning(f"Target data CSV file does not exist: {csv_path}")
                logger.warning("Will continue processing, generating IDs and metadata")
        
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
            logger.info(f"Generated automatic IDs and metadata for {len(auto_ids)} target embeddings")
        
        # 将嵌入向量添加到DataFrame
        embedding_lists = [row.tolist() for row in embeddings]
        df_result = df_csv.with_columns(pl.Series("embedding", embedding_lists))
        
        # 如果id列中有空值，进行过滤
        if df_result["id"].is_null().any():
            null_count = df_result["id"].is_null().sum()
            logger.warning(f"Found {null_count} null IDs in target data, which will be filtered out")
            df_result = df_result.filter(pl.col("id").is_not_null())
        
        logger.info(f"Successfully loaded target data, shape: {df_result.shape}")
        return df_result
    except Exception as e:
        logger.exception(f"Error loading target data: {e}")
        return None


def prepare_training_data(df_prefs: pl.DataFrame, df_bg: pl.DataFrame, neg_ratio: float, random_state: int, 
                        oversample_method: str = "borderline-smote", oversample_ratio: float = 0.5, 
                        confidence_weighted: bool = True, initial_model = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Prepare training data by sampling negative samples from the background, implementing adaptive boundary-focused oversampling and confidence-weighted positive sampling.
    
    Note: Will automatically filter out samples with empty IDs
    
    Args:
        df_prefs: Preference data DataFrame containing id, embedding, label columns
        df_bg: Background data DataFrame containing id, embedding columns
        neg_ratio: Ratio of negative to positive samples
        random_state: Random seed
        oversample_method: Oversampling method, options: "none", "borderline-smote", "adasyn"
        oversample_ratio: Oversampling ratio, number of synthetic samples = original positive sample count * this ratio
        confidence_weighted: Whether to perform confidence-weighted sampling for positive samples
        initial_model: Model used to calculate sample confidence, if None and confidence_weighted is True, a simple model will be trained
    
    Returns:
        Processed feature matrix and label vector, or None if processing fails
    """
    
    logger.info(f"Preparing training data, configuration: neg_ratio={neg_ratio}, oversample_method={oversample_method}, oversample_ratio={oversample_ratio}, confidence_weighted={confidence_weighted}")
    
    # Filter out samples with empty IDs
    if df_prefs["id"].is_null().any():
        null_count = df_prefs["id"].is_null().sum()
        logger.warning(f"Found {null_count} null IDs in preference data, which will be filtered out")
        df_prefs = df_prefs.filter(pl.col("id").is_not_null())
        
        if df_prefs.is_empty():
            logger.error("After filtering null IDs, preference data is empty. Cannot train.")
            return None
    
    # Handle null embeddings
    if df_prefs["embedding"].is_null().any():
        null_count = df_prefs["embedding"].is_null().sum()
        logger.warning(f"Found {null_count} null embeddings in preference data, which will be filtered out")
        df_prefs = df_prefs.filter(pl.col("embedding").is_not_null())
        
        if df_prefs.is_empty():
            logger.error("After filtering null embeddings, preference data is empty. Cannot train.")
            return None
    
    # Separate positive and negative samples
    df_pos = df_prefs.filter(pl.col('label') == 1)
    df_neg_explicit = df_prefs.filter(pl.col('label') == 0)
    
    n_positive = df_pos.height
    n_neg_explicit = df_neg_explicit.height
    
    if n_positive == 0:
        logger.error("No positive samples found. Cannot train.")
        return None

    # Remove preference IDs from background pool
    pref_ids = set(df_prefs['id'].to_list())
    df_bg_pool = df_bg.filter(pl.col('id').is_in(pref_ids).not_())
    n_available_bg = df_bg_pool.height
    
    if n_available_bg == 0:
        logger.warning("No available background samples left after removing preference IDs.")
        n_to_sample = 0
    else:
        # Sample neg_ratio times the number of positive samples
        n_to_sample = int(n_positive * neg_ratio)
        n_to_sample = min(n_to_sample, n_available_bg)  # Cannot exceed available background samples
        logger.info(f"Sampling logic - Positive: {n_positive}, Explicit Negative: {n_neg_explicit}, Sampled Negative: {n_to_sample} (from {n_available_bg} background samples)")

    # Sample negative samples
    if n_to_sample > 0:
        df_neg_sampled = df_bg_pool.sample(n=n_to_sample, shuffle=True, seed=random_state)
        df_neg_sampled = df_neg_sampled.with_columns(pl.lit(0).alias('label'))
        df_neg_sampled = df_neg_sampled.select(["id", "embedding", "label"]) 
    else:
        df_neg_sampled = pl.DataFrame({"id": [], "embedding": [], "label": []}, schema={"id": pl.Utf8, "embedding": pl.List(pl.Float32), "label": pl.Int8})

    # Combine original data - positive samples will be weighted later
    df_train_initial = pl.concat([
        df_pos.select(["id", "embedding", "label"]), 
        df_neg_explicit.select(["id", "embedding", "label"]), 
        df_neg_sampled
    ], how="vertical_relaxed")

    # Final check
    if df_train_initial.is_empty():
        logger.error("Processed training data is empty.")
        return None
        
    # --- Implement confidence-weighted sampling ---
    if confidence_weighted and initial_model is None:
        logger.info("No initial model provided, training a simple model for confidence calculation")
        try:
            # Prepare temporary dataset to train initial model
            X_temp = np.array(df_train_initial['embedding'].to_list(), dtype=np.float64)
            y_temp = df_train_initial['label'].to_numpy()
            
            # Check data validity
            if np.isnan(X_temp).any() or np.isinf(X_temp).any():
                logger.warning("Temporary data contains NaN or Inf values, skipping confidence-weighted sampling")
                confidence_weighted = False
            else:
                # Train a simple logistic regression model
                initial_model = LogisticRegression(C=1.0, max_iter=500, random_state=random_state, class_weight='balanced')
                initial_model.fit(X_temp, y_temp)
                logger.info("Successfully trained initial model for confidence calculation")
        except Exception as e:
            logger.warning(f"Error training initial model, skipping confidence-weighted sampling: {e}")
            confidence_weighted = False
    
    # Perform confidence-weighted sampling
    if confidence_weighted and initial_model is not None:
        try:
            # Calculate the number of positive samples needed, keep the same as before
            pos_sample_count = df_pos.height
            
            # Perform confidence-weighted sampling
            logger.info("Performing confidence-weighted positive sampling...")
            df_pos_weighted = confidence_weighted_sampling(
                df_pos, 
                initial_model,
                high_conf_threshold=0.9,  # High confidence threshold
                high_conf_weight=2.0,     # Weight for high confidence samples
                n_samples=pos_sample_count,
                random_state=random_state
            )
            
            # Recombine data, replace original positive samples with weighted ones
            df_train = pl.concat([
                df_pos_weighted, 
                df_neg_explicit.select(["id", "embedding", "label"]), 
                df_neg_sampled
            ], how="vertical_relaxed")
            
            logger.info("Completed confidence-weighted sampling, replaced original positive samples")
        except Exception as e:
            logger.warning(f"Error during confidence-weighted sampling: {e}, will use original data")
            df_train = df_train_initial
    else:
        df_train = df_train_initial
        if confidence_weighted:
            logger.info("Skipping confidence-weighted sampling, using original positive samples")
    
    # Convert DataFrame to NumPy arrays
    try:
        # Ensure embeddings are not null
        df_train = df_train.filter(pl.col('embedding').is_not_null())
        if df_train.is_empty():
             logger.error("After filtering null embeddings, training data is empty.")
             return None
             
        X = np.array(df_train['embedding'].to_list(), dtype=np.float64)
        y = df_train['label'].to_numpy()

        # Check for NaN/Inf
        if np.isnan(X).any():
            nan_rows = np.isnan(X).any(axis=1)
            logger.error(f"NaN values found in training features, {np.sum(nan_rows)} rows")
            return None
        if np.isinf(X).any():
            inf_rows = np.isinf(X).any(axis=1)
            logger.error(f"Infinite values found in training features, {np.sum(inf_rows)} rows")
            return None
            
        # --- Implement boundary-focused oversampling ---
        if oversample_method.lower() not in ["none", "no", "false", ""]:
            logger.info(f"Executing boundary-focused oversampling, method: {oversample_method}")
            
            if oversample_method.lower() == "borderline-smote":
                # Use Borderline-SMOTE for oversampling
                X, y = borderline_smote(
                    X, y,
                    n_neighbors=5,
                    sampling_ratio=oversample_ratio,
                    random_state=random_state
                )
            elif oversample_method.lower() == "adasyn":
                # Use ADASYN for oversampling
                X, y = adasyn(
                    X, y,
                    beta=oversample_ratio,
                    n_neighbors=5,
                    random_state=random_state
                )
            else:
                logger.warning(f"Unsupported oversampling method: {oversample_method}, skipping oversampling")
            
            # Log class distribution after oversampling
            pos_count_after = np.sum(y == 1)
            neg_count_after = np.sum(y == 0)
            logger.info(f"Class distribution after oversampling - Positive: {pos_count_after}, Negative: {neg_count_after}, Ratio: {neg_count_after/pos_count_after:.2f}:1")
        else:
            logger.info("Skipping boundary-focused oversampling")

        logger.info(f"Prepared training features shape: X: {X.shape}, y: {y.shape}")
        return X, y
    except Exception as e:
        logger.exception(f"Error converting training data to NumPy arrays or performing oversampling: {e}")
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

        # Use logistic regression model
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
        logger.info(f"Training set positive ratio: {np.mean(y_train):.3f}, Test set positive ratio: {np.mean(y_test):.3f}") # loguru output

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


def adaptive_sample(scores: np.ndarray, target_rate: float = 0.15, 
                    high_percentile: float = 90, boundary_percentile: float = 50,
                    random_state: int = 42) -> np.ndarray:
    """Adaptive boundary sampling, dynamically determines threshold and controls overall sampling rate.
    
    Args:
        scores: Array of scores predicted by the model
        target_rate: Target overall sampling rate (small decimal between 0-1, e.g. 0.15 means select 15%)
        high_percentile: High confidence score percentile (e.g. 90 means top 10% of scores are high confidence)
        boundary_percentile: Lower bound percentile for boundary region
        random_state: Random seed
        
    Returns:
        Boolean mask array indicating whether each sample is selected
    """
    np.random.seed(random_state)
    n_samples = len(scores)
    target_count = int(n_samples * target_rate)
    
    logger.info(f"Performing adaptive boundary sampling: target rate={target_rate:.2f} (target count={target_count}), "
                f"high confidence percentile={high_percentile}, boundary percentile={boundary_percentile}")
    
    # Dynamically determine high confidence threshold
    high_threshold = np.percentile(scores, high_percentile)
    
    # Select all high confidence samples
    high_indices = np.where(scores >= high_threshold)[0]
    show = np.zeros(n_samples, dtype=bool)
    show[high_indices] = True
    high_count = len(high_indices)
    
    logger.info(f"High confidence threshold: {high_threshold:.4f}, selected {high_count} high confidence samples ({high_count/n_samples*100:.1f}%)")
    
    # If high confidence samples already meet or exceed target count, adjust selection
    if high_count >= target_count:
        # May need to randomly select a portion of high confidence samples
        if high_count > target_count:
            logger.info(f"High confidence sample count ({high_count}) exceeds target count ({target_count}), will randomly select {target_count} samples")
            # Randomly select target_count high confidence samples
            selected_indices = np.random.choice(high_indices, target_count, replace=False)
            show = np.zeros(n_samples, dtype=bool)
            show[selected_indices] = True
    else:
        # Define boundary region and sample remaining needed samples from it
        remaining = target_count - high_count
        boundary_threshold = np.percentile(scores, boundary_percentile)
        
        logger.info(f"High confidence sample count ({high_count}) is less than target count ({target_count}), "
                   f"will sample additional {remaining} samples from boundary region")
        logger.info(f"Boundary region threshold: {boundary_threshold:.4f} ~ {high_threshold:.4f}")
        
        # Get samples from boundary region
        boundary_indices = np.where((scores >= boundary_threshold) & (scores < high_threshold))[0]
        
        if len(boundary_indices) > 0:
            # Calculate weights for boundary samples (higher weight for closer to high threshold)
            boundary_scores = scores[boundary_indices]
            
            # Calculate normalized weights
            min_score = boundary_scores.min()
            score_range = boundary_scores.max() - min_score
            
            if score_range > 1e-6:  # Avoid division by zero error
                weights = (boundary_scores - min_score) / score_range
            else:
                weights = np.ones_like(boundary_scores)
                
            # Enhance score differences to make high scores more likely to be selected (optional)
            weights = np.power(weights, 2)  # Square weights to amplify differences
            
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Sample from boundary region, not exceeding remaining needed count
            sample_size = min(remaining, len(boundary_indices))
            
            if sample_size > 0:
                sampled_boundary = np.random.choice(
                    boundary_indices, 
                    size=sample_size, 
                    replace=False, 
                    p=weights
                )
                show[sampled_boundary] = True
                
                # Record sampling statistics
                sampled_scores = scores[sampled_boundary]
                logger.info(f"Sampled {sample_size} samples from boundary region, average score: {np.mean(sampled_scores):.4f}")
        else:
            logger.warning(f"No available samples in boundary region (scores >= {boundary_threshold:.4f} and < {high_threshold:.4f})")
    
    total_shown = np.sum(show)
    logger.info(f"Finally marked {total_shown} / {n_samples} ({total_shown/n_samples*100:.1f}%) samples for display")
    
    return show.astype(np.uint8)  # Return in 0/1 format


def display_sample_papers(df: pl.DataFrame, threshold: float, n_samples: int = 5) -> None:
    """Display sample papers from different score categories.

    Args:
        df: DataFrame containing paper info and predicted scores
        threshold: Model prediction threshold
        n_samples: Number of samples to display from each category
    """
    if n_samples <= 0 or df.is_empty():
        return

    # Ensure required columns exist
    required_cols = ['id', 'score']
    optional_cols = ['title', 'abstract', 'authors', 'date', 'primary_category']
    
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns, cannot display samples. Required: {required_cols}")
        return
    
    available_info_cols = [col for col in optional_cols if col in df.columns]
    
    # Define score ranges
    high_scores = df.filter(pl.col('score') >= threshold + 0.2)
    medium_scores = df.filter((pl.col('score') >= threshold - 0.1) & 
                            (pl.col('score') < threshold + 0.1))
    low_scores = df.filter(pl.col('score') < threshold - 0.2)
    
    # Sample and display
    for category, df_category, name in [
        ("High Score Papers", high_scores, "HIGH"), 
        ("Medium Score Papers", medium_scores, "MEDIUM"), 
        ("Low Score Papers", low_scores, "LOW")
    ]:
        if not df_category.is_empty():
            samples = df_category.sample(n=min(n_samples, df_category.height), shuffle=True, seed=42)
            logger.info(f"\n{'='*50}\n{name} SCORE PAPERS\n{'='*50}")
            
            for row in samples.sort(pl.col('score'), descending=True).rows(named=True):
                logger.info(f"\nPaper ID: {row['id']} - Score: {row['score']:.4f}")
                
                if 'title' in row and row['title']:
                    logger.info(f"Title: {row['title']}")
                
                if 'abstract' in row and row['abstract']:
                    abstract = row['abstract']
                    # Truncate long abstracts
                    if len(abstract) > 500:
                        abstract = abstract[:500] + "..."
                    logger.info(f"Abstract: {abstract}")
                
                if 'authors' in row and row['authors']:
                    logger.info(f"Authors: {row['authors']}")
                
                if 'date' in row and row['date']:
                    logger.info(f"Published Date: {row['date']}")
                
                if 'primary_category' in row and row['primary_category']:
                    logger.info(f"Primary Category: {row['primary_category']}")
                
                # Add separator line
                logger.info("-" * 30)


# --- Main Script Logic ---
def main():
    # Configure Chinese font support
    # configure_chinese_font()
    
    # Load the entire config
    full_config = load_config()
    # Get specific sections, defaulting to empty dict if section is missing
    model_fitting_cfg = full_config.get('model_fitting', {})

    parser = argparse.ArgumentParser(description="Train preference model, predict scores, and perform sampling. Reads defaults from config.toml [model_fitting].")
    # Updated arguments to read defaults from model_fitting_cfg
    parser.add_argument("--preference-dir", "-p", type=str, 
                        default=model_fitting_cfg.get('preference_dir'),
                        help="Directory containing preference CSV+NPY files.")
    parser.add_argument("--background-file", "-b", type=str, 
                        default=model_fitting_cfg.get('background_file'),
                        help="Background data file path, can be CSV or NPY file. If NPY file provided, IDs will be generated.")
    parser.add_argument("--target-file", "-t", type=str, 
                        default=model_fitting_cfg.get('target_file'),
                        help="Target prediction file path, can be CSV or NPY file. If NPY file provided, IDs and metadata will be generated.")
    parser.add_argument("--neg-ratio", type=float, 
                        default=model_fitting_cfg.get('neg_ratio', 5.0),
                        help="Target ratio of negative to positive samples after sampling.")
    parser.add_argument("--folds", "-k", type=int, 
                        default=model_fitting_cfg.get('folds', 5),
                        help="Number of folds for cross-validation.")
    parser.add_argument("--beta", type=float, 
                        default=model_fitting_cfg.get('beta', 1.2),
                        help="Beta value for F-beta/G-mean optimization in threshold finding.")
    parser.add_argument("--random-state", type=int, 
                        default=model_fitting_cfg.get('random_state', 42),
                        help="Random state for reproducibility.")
    # Add sample papers argument
    parser.add_argument("--sample", type=int,
                        default=model_fitting_cfg.get('sample', 0),
                        help="Number of sample papers to display from each score category (0 to disable).")
    # Add visualization directory argument
    parser.add_argument("--visualization-dir", type=str,
                        default=model_fitting_cfg.get('visualization_dir'),
                        help="Directory to save visualization plots")
    parser.add_argument("--no-visualization", action="store_true",
                        help="Disable generating visualizations")
    
    # Add adaptive boundary-focused oversampling arguments
    parser.add_argument("--oversample-method", type=str,
                        default=model_fitting_cfg.get('oversample_method', 'borderline-smote'),
                        choices=['none', 'borderline-smote', 'adasyn'],
                        help="Boundary-focused oversampling method: none(no oversampling), borderline-smote, adasyn")
    parser.add_argument("--oversample-ratio", type=float,
                        default=model_fitting_cfg.get('oversample_ratio', 0.5),
                        help="Oversampling ratio, number of synthetic samples = original positive sample count * this ratio")
    
    # Add confidence-weighted sampling arguments
    parser.add_argument("--confidence-weighted", action="store_true",
                        default=model_fitting_cfg.get('confidence_weighted', True),
                        help="Enable confidence-weighted positive sampling")
    parser.add_argument("--high-conf-threshold", type=float,
                        default=model_fitting_cfg.get('high_conf_threshold', 0.9),
                        help="High confidence sample threshold")
    parser.add_argument("--high-conf-weight", type=float,
                        default=model_fitting_cfg.get('high_conf_weight', 2.0),
                        help="Weight multiplier for high confidence samples")
    
    # Add sampling-related arguments
    parser.add_argument("--target-sample-rate", type=float,
                        default=model_fitting_cfg.get('target_sample_rate', 0.15),
                        help="Target sampling rate (0-1 decimal), controls final recommendation count")
    parser.add_argument("--high-percentile", type=float,
                        default=model_fitting_cfg.get('high_percentile', 90),
                        help="High confidence threshold percentile, higher value means stricter filtering")
    parser.add_argument("--boundary-percentile", type=float,
                        default=model_fitting_cfg.get('boundary_percentile', 50),
                        help="Lower bound percentile for boundary region")

    args = parser.parse_args()

    # --- Set Random Seeds for Reproducibility ---
    seed = args.random_state
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seeds (python, numpy) to: {seed}")

    # --- Create Plotting Directory (if enabled) ---
    plots_enabled = not args.no_visualization
    plots_dir = None
    if plots_enabled and args.visualization_dir:
        plots_dir = pathlib.Path(args.visualization_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {plots_dir}")

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
    # Call prepare_training_data function with updated parameters
    training_data = prepare_training_data(
        df_prefs, 
        df_bg, 
        args.neg_ratio, 
        args.random_state,
        oversample_method=args.oversample_method,
        oversample_ratio=args.oversample_ratio,
        confidence_weighted=args.confidence_weighted,
        initial_model=None  # Initial model set to None, function will create one if needed
    )
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
    
    # Use fixed threshold instead of automatically calculated threshold
    fixed_threshold = 0.5
    logger.info(f"Automatically calculated optimal threshold: {optimal_threshold:.4f}, but will use fixed threshold: {fixed_threshold}")

    # --- Final Model Training ---
    logger.info("--- Training Final Model --- ")
    # Use logistic regression model
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

    # --- Predict scores on Background Data ---
    logger.info("--- Predicting scores on Background Data ---")
    try:
        X_bg = np.array(df_bg['embedding'].to_list(), dtype=np.float32)
        
        if np.isnan(X_bg).any() or np.isinf(X_bg).any():
            logger.warning("NaN or Inf values found in background data, may affect prediction quality")
        
        bg_scores = final_model.predict_proba(X_bg)[:, 1]
        df_bg = df_bg.with_columns(pl.Series("score", bg_scores))
        logger.info(f"Successfully calculated prediction scores for {len(bg_scores)} background samples")
    except Exception as e:
        logger.exception("Error predicting scores on background data")
        bg_scores = np.array([])

    # --- Get Training Data Scores ---
    logger.info("--- Getting Training Data Scores ---")
    try:
        train_scores = final_model.predict_proba(X_train)[:, 1]
        logger.info(f"Successfully obtained prediction scores for {len(train_scores)} training samples")
        
        # Separate positive and negative sample scores
        pos_scores = train_scores[y_train == 1]
        neg_scores = train_scores[y_train == 0]
        logger.info(f"Training set: {len(pos_scores)} positive samples, {len(neg_scores)} negative samples")
    except Exception as e:
        logger.exception("Error getting training data scores")
        train_scores = np.array([])
        pos_scores = np.array([])
        neg_scores = np.array([])

    # --- Generate Visualizations ---
    if not args.no_visualization and args.visualization_dir:
        logger.info("--- Generating Visualizations ---")
        visualization_dir = pathlib.Path(args.visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot score distributions
        try:
            scores_dict = {
                "Target Data": target_scores,
                "Background Data": bg_scores,
                "Training Data (Positive Samples)": pos_scores
            }
            
            scores_path = visualization_dir / f"score_distributions_{timestamp}.png"
            plot_score_distributions(scores_dict, fixed_threshold, str(scores_path))
            logger.info("Score distribution plot generated")
        except Exception as e:
            logger.exception("Error generating score distribution plot")
        
        # Plot ROC curve
        try:
            roc_path = visualization_dir / f"roc_curve_{timestamp}.png"
            plot_roc_curve(cv_true, cv_probs, fixed_threshold, args.beta, str(roc_path))
            logger.info("ROC curve plot generated")
        except Exception as e:
            logger.exception("Error generating ROC curve plot")

    # --- Biased Sampling ---
    logger.info("--- Performing Adaptive Sampling ---")
    
    # Use adaptive sampling strategy
    logger.info(f"Using adaptive sampling strategy, target sample rate: {args.target_sample_rate}, " 
                f"high confidence percentile: {args.high_percentile}, boundary percentile: {args.boundary_percentile}")
    show_flags = adaptive_sample(
        target_scores, 
        target_rate=args.target_sample_rate,
        high_percentile=args.high_percentile,
        boundary_percentile=args.boundary_percentile,
        random_state=args.random_state
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

    # --- Display Sample Papers ---
    if args.sample > 0:
        logger.info(f"--- Displaying Sample Papers ({args.sample} from each category) ---")
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