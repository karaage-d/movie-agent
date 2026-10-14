# -*- coding: utf-8 -*-
"""
工具模块：配置加载、通用工具
"""
import os
import yaml
import logging
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent


def setup_model_cache():
    """把 HuggingFace / sentence-transformers / torch 的缓存目录锁定到项目内的 models_cache/。

    这些库默认下载到 C:\\Users\\<用户>\\.cache\\huggingface，与项目分离；统一指向项目目录后
    模型随项目走。用 setdefault，外部显式设置的同名环境变量优先。
    """
    cache_root = ROOT_DIR / "models_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_root / "sentence_transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    # 直连 huggingface.co 容易超时，默认走镜像，有代理可在外部覆盖
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# 必须在 import huggingface_hub / transformers / sentence_transformers 之前执行
setup_model_cache()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("movie-agent")


def load_config(config_path=None):
    """加载YAML配置文件"""
    if config_path is None:
        config_path = ROOT_DIR / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


# 中文评分描述
def rating_label(score, scale=10.0):
    ratio = score / scale
    if ratio >= 0.9: return "神作 🏆"
    if ratio >= 0.8: return "优秀 🌟"
    if ratio >= 0.7: return "良好 👍"
    if ratio >= 0.6: return "一般 😐"
    if ratio >= 0.5: return "较差 👎"
    return "烂片 💩"


def is_blank(v):
    """判断值是否为空 / NaN / NaT / 'nan' 字符串，供多处共用。"""
    import math

    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    try:
        import pandas as pd

        if v is pd.NaT:
            return True
        if pd.api.types.is_scalar(v):
            try:
                if pd.isna(v):
                    return True
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    text = str(v).strip()
    return text == "" or text.lower() in ("nan", "nat", "none", "null")


def init_dirs(cfg):
    for key in ["raw_dir", "cleaned_dir", "posters_dir"]:
        ensure_dir(ROOT_DIR / cfg["data"][key])
    ensure_dir(ROOT_DIR / cfg["vectorstore"]["persist_dir"])
    ensure_dir(ROOT_DIR / cfg["analysis"]["output_dir"])
    ensure_dir(ROOT_DIR / cfg["embedding"]["cache_dir"])
