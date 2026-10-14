# -*- coding: utf-8 -*-
"""电影查询智能体 HTTP API 服务。

只用 Python 标准库（http.server / json / threading 等），不依赖任何 Web 框架。

运行（在项目根目录或任意目录均可）：
    python server/app.py

服务地址： http://127.0.0.1:8765

实现要点：
  1. 启动时用后台线程初始化 LLM 与知识库，HTTP 服务立即可用；
  2. 初始化进度与失败原因通过 GET /api/status 暴露，初始化失败不会让服务退出；
  3. ThreadingHTTPServer，聊天请求不阻塞其它请求；
  4. MovieAgent 是共享单例且带多轮对话历史，chat 调用需在锁内串行；
  5. 响应统一为 application/json; charset=utf-8（图片接口除外），NaN/NaT 转 null。
"""
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

# server/app.py -> 项目根目录。与 main.py 一样先插入 sys.path，
# 保证从任意工作目录启动时 `from src.xxx import ...` 都能解析。
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 先锁定模型缓存目录（项目内 models_cache/），再导入任何 HF 相关库
from src.utils import load_config, setup_model_cache  # noqa: E402

setup_model_cache()

HOST = "127.0.0.1"
PORT = 8765

WEB_DIR = ROOT_DIR / "web"
POSTER_DIR = ROOT_DIR / "data" / "posters"
ANALYSIS_DIR = ROOT_DIR / "output" / "analysis"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
}

CORS_HEADERS = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With"),
    ("Access-Control-Max-Age", "86400"),
]

# 电影标题 / 文件名里需要替换掉的 Windows 非法字符
ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')

MAX_BODY_BYTES = 1 << 20  # 1MB，足够聊天输入
CHAT_TIMEOUT_SEC = 90.0
ANALYSIS_CACHE_SEC = 600.0


# 通用小工具
def json_default(obj):
    """让 numpy / pandas 的标量类型可以被 json 序列化。"""
    try:
        item = obj.item()
        return item
    except Exception:
        return str(obj)


def dumps(obj):
    """统一出口：ensure_ascii=False，NaN/Inf 一律变成 null。"""
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=json_default,
        allow_nan=False,
    ).encode("utf-8")


def exc_text(exc):
    """异常 -> 可读的错误字符串。"""
    msg = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def is_blank(v):
    """判断值是否为空 / NaN / NaT / 'nan' 字符串。"""
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


def to_str(v):
    """任意值 -> 字符串（NaN/NaT -> ''）。"""
    if is_blank(v):
        return ""
    return str(v).strip()


def to_int(v):
    """任意值 -> int 或 None。"""
    if is_blank(v):
        return None
    try:
        f = float(str(v).strip())
        if math.isnan(f) or math.isinf(f):
            return None
        return int(round(f))
    except (TypeError, ValueError):
        return None


def to_float(v, ndigits=None):
    """任意值 -> float 或 None，可指定位数。"""
    if is_blank(v):
        return None
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, ndigits) if ndigits is not None else f


def split_list(raw):
    """把 '科幻/灾难,冒险' 拆成 ['科幻','灾难','冒险']。"""
    if is_blank(raw):
        return []
    parts = re.split(r"[/,，、;；|]", str(raw))
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]


# 应用状态（全局单例）
class AppState:
    """进程级共享状态：配置、数据、模型、分析缓存。"""

    def __init__(self):
        self.lock = threading.Lock()          # 保护 agent 串行调用
        self.init_lock = threading.Lock()     # 保护初始化流程

        self.cfg = None
        self.movies_df = None
        self.reviews_df = None
        self.review_stats = {}                # movie_id -> {"count": int, "good_rate": float|None}
        self.title_cache = []                 # [(title, title_lower)] 按长度降序

        self.llm = None
        self.emb = None
        self.kb = None
        self.agent = None

        self.initializing = False
        self.ready = False
        self.error = None
        self.model_name = ""
        self.embed_name = ""
        self.poster_stats = {"have": 0, "total": 0}

        self._analysis_cache = None
        self._analysis_cache_at = 0.0
        self._analysis_lock = threading.Lock()

        self.good_threshold = 7.0

    # 状态
    def status(self):
        with self.init_lock:
            return {
                "ready": bool(self.ready),
                "initializing": bool(self.initializing),
                "movies": int(len(self.movies_df)) if self.movies_df is not None else 0,
                "reviews": int(len(self.reviews_df)) if self.reviews_df is not None else 0,
                "model": self.model_name or "",
                "embedding": self.embed_name or "",
                "poster": dict(self.poster_stats),
                "error": self.error,
            }

    # 配置
    def load_config_once(self):
        """加载配置 + CSV 数据 + 快速统计，返回是否成功。"""
        if self.cfg is not None:
            return True
        try:
            self.cfg = load_config()
        except Exception as exc:
            self.set_error(f"配置加载失败: {exc_text(exc)}")
            return False

        self.model_name = self._model_label(self.cfg)
        self.embed_name = self._embed_label(self.cfg)
        try:
            self.good_threshold = float(self.cfg["agent"]["good_rating_threshold"])
        except Exception:
            self.good_threshold = 7.0

        try:
            self.load_data()
        except Exception as exc:
            # 数据加载失败不阻塞服务，只有依赖数据的接口会报错
            self.set_error(f"数据加载失败: {exc_text(exc)}")
            return True
        return True

    @staticmethod
    def _model_label(cfg):
        """当前对话模型的可读名称：跟随 provider 取对应模型名，顺便带上供应商。"""
        try:
            m = cfg["model"]
            provider = str(m["provider"]).lower()
            if provider == "qwen":
                return str(m.get("qwen_model", "qwen"))
            if provider == "deepseek":
                return str(m.get("deepseek_model", "deepseek"))
            if provider == "ollama":
                return str(m.get("ollama_model", "ollama"))
            if provider == "openai":
                return str(m.get("openai_model", "openai"))
            return provider
        except Exception:
            return ""

    @staticmethod
    def _embed_label(cfg):
        """当前向量模型的名称，供前端展示（bge 走本地，qwen 走在线）。"""
        try:
            return str(cfg["embedding"].get("model_name", ""))
        except Exception:
            return ""

    def load_data(self):
        """直接读 CSV（不依赖 LLM），用于 /api/status 与列表接口。"""
        import pandas as pd

        movies_path = ROOT_DIR / self.cfg["data"]["movies_csv"]
        reviews_path = ROOT_DIR / self.cfg["data"]["reviews_csv"]

        if not movies_path.exists():
            raise FileNotFoundError(f"电影数据不存在: {movies_path}")

        self.movies_df = pd.read_csv(movies_path, encoding="utf-8-sig")
        self.reviews_df = (
            pd.read_csv(reviews_path, encoding="utf-8-sig") if reviews_path.exists() else pd.DataFrame()
        )

        # 预计算每部电影的好评率 / 评论数
        stats = {}
        if self.reviews_df is not None and len(self.reviews_df) > 0 and "rating" in self.reviews_df.columns:
            rdf = self.reviews_df.dropna(subset=["rating"])
            for mid, group in rdf.groupby("movie_id"):
                total = int(len(group))
                if total == 0:
                    continue
                good = int((group["rating"].astype(float) >= self.good_threshold).sum())
                stats[int(mid)] = {"count": total, "good_rate": round(good / total * 100, 1)}
        self.review_stats = stats

        titles = [to_str(t) for t in self.movies_df["title"].tolist()]
        self.title_cache = sorted(
            [(t, t.lower()) for t in titles if t],
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        self.poster_stats = self._count_posters(titles)

    @staticmethod
    def _count_posters(titles):
        """统计本地 data/posters 缓存覆盖率，供前端区分「没有海报数据」和「加载失败」。"""
        try:
            have = sum(1 for t in titles if find_poster_file(t))
            return {"have": int(have), "total": int(len(titles))}
        except Exception:
            return {"have": 0, "total": 0}

    def set_error(self, message):
        with self.init_lock:
            self.error = message
        print(f"[server] ERROR: {message}", flush=True)

    # 初始化
    def start_init(self):
        """后台线程加载 LLM + 知识库 + Agent，不阻塞 HTTP 服务。"""
        if not self.load_config_once():
            return
        with self.init_lock:
            if self.initializing or self.ready:
                return
            self.initializing = True
            self.error = None
        thread = threading.Thread(target=self._init_worker, name="movie-agent-init", daemon=True)
        thread.start()

    def _init_worker(self):
        try:
            from src.model_loader import load_embedding, load_llm

            print("[server] 正在加载 LLM ...", flush=True)
            self.llm = load_llm(self.cfg)
            print("[server] LLM 加载完成，正在加载 Embedding ...", flush=True)
            self.emb = load_embedding(self.cfg)
            print("[server] Embedding 加载完成，正在构建/加载知识库 ...", flush=True)

            from src.knowledge_base import MovieKnowledgeBase

            self.kb = MovieKnowledgeBase(self.cfg, self.emb).build()
            print("[server] 知识库就绪，正在创建 Agent ...", flush=True)

            from src.agent import MovieAgent

            self.agent = MovieAgent(self.cfg, self.llm, self.kb)

            # 知识库可能重新读了 CSV，同步一份最新数据
            if getattr(self.kb, "movies_df", None) is not None:
                self.movies_df = self.kb.movies_df
            if getattr(self.kb, "reviews_df", None) is not None and len(self.kb.reviews_df) > 0:
                self.reviews_df = self.kb.reviews_df

            with self.init_lock:
                self.ready = True
                self.error = None
            print("[server] ✅ 初始化完成，服务就绪", flush=True)
        except Exception as exc:
            traceback.print_exc()
            self.set_error(f"初始化失败: {exc_text(exc)}")
        finally:
            with self.init_lock:
                self.initializing = False

    # 数据辅助
    def require_data(self):
        if self.cfg is None:
            self.load_config_once()
        if self.movies_df is None or self.cfg is None:
            raise RuntimeError("电影数据尚未加载完成，请稍候")
        return self.movies_df

    def review_stat(self, movie_id):
        return self.review_stats.get(int(movie_id), {"count": 0, "good_rate": None})

    def poster_url(self, title):
        path = find_poster_file(title)
        if path is None:
            # 本地没有这张海报时返回 None，前端才能显示「暂无海报」而不是裂图
            return None
        return f"/api/poster?title={quote(title, safe='')}"

    def movie_item(self, row):
        """DataFrame 行 -> 前端契约对象（NaN 全部变 null）。"""
        mid = to_int(row.get("movie_id")) or 0
        title = to_str(row.get("title"))
        stat = self.review_stat(mid)
        return {
            "movie_id": mid,
            "title": title,
            "year": to_int(row.get("year")),
            "director": to_str(row.get("director")),
            "actors": to_str(row.get("actors")),
            "genre": to_str(row.get("genre")),
            "douban_rating": to_float(row.get("douban_rating")),
            "maoyan_rating": to_float(row.get("maoyan_rating")),
            "imdb_rating": to_float(row.get("imdb_rating")),
            "duration": to_int(row.get("duration")),
            "region": to_str(row.get("region")),
            "language": to_str(row.get("language")),
            "box_office": to_str(row.get("box_office")),
            "summary": to_str(row.get("summary")),
            "poster": self.poster_url(title),
            "good_rate": stat["good_rate"],
            "review_count": int(stat["count"]),
        }

    def find_movie_row(self, title):
        """包含匹配，等价于 knowledge_base.get_movie_info 的行为。"""
        df = self.require_data()
        kw = to_str(title)
        if not kw:
            return None
        matches = df[df["title"].astype(str).str.contains(kw, na=False, case=False, regex=False)]
        if len(matches) == 0:
            return None
        return matches.iloc[0]

    def extract_movie_title(self, *texts):
        """从文本里尽力抽出电影名（能匹配到 movies.csv 标题就返回标题）。"""
        for text in texts:
            if not text:
                continue
            low = str(text).lower()
            for title, title_low in self.title_cache:
                if title_low and title_low in low:
                    return title
        return None

    def get_analysis(self):
        """分析结果 + 图表，缓存 10 分钟，避免每次重算重画。"""
        import pandas as pd

        now = time.time()
        with self._analysis_lock:
            if self._analysis_cache is not None and (now - self._analysis_cache_at) < ANALYSIS_CACHE_SEC:
                return self._analysis_cache

        from src.analyzer import MovieCorrelationAnalyzer

        results = MovieCorrelationAnalyzer(self.cfg).analyze() or {}

        insights = [str(x) for x in (results.get("insights") or [])]

        correlations = []
        for item in results.get("strong_correlations") or []:
            correlations.append(
                {
                    "var1": to_str(item.get("var1")),
                    "var2": to_str(item.get("var2")),
                    "correlation": to_float(item.get("correlation")) or 0.0,
                    "direction": to_str(item.get("direction")),
                }
            )

        genres = []
        gdf = results.get("genre_analysis")
        if isinstance(gdf, pd.DataFrame) and not gdf.empty:
            for _, row in gdf.iterrows():
                genres.append(
                    {
                        "genre": to_str(row.get("genre")),
                        "count": to_int(row.get("count")) or 0,
                        "avg_rating": to_float(row.get("avg_rating")),
                    }
                )

        directors = []
        ddf = results.get("top_directors")
        if isinstance(ddf, pd.DataFrame) and not ddf.empty:
            for _, row in ddf.iterrows():
                directors.append(
                    {
                        "director": to_str(row.get("director")),
                        "count": to_int(row.get("count")) or 0,
                        "avg_rating": to_float(row.get("avg_rating")),
                    }
                )

        payload = {
            "insights": insights,
            "correlations": correlations,
            "genres": genres,
            "directors": directors,
            "charts": {"boxplot": "/api/chart/boxplot"},
        }

        with self._analysis_lock:
            self._analysis_cache = payload
            self._analysis_cache_at = time.time()
        return payload


APP = AppState()


# 海报文件查找
def _safe_name(title):
    """标题 -> 文件名（Windows 非法字符替换为 _）。"""
    return ILLEGAL_FILENAME_CHARS.sub("_", str(title)).strip()


def _poster_files():
    if not POSTER_DIR.is_dir():
        return []
    try:
        return sorted(POSTER_DIR.iterdir(), key=lambda p: p.name)
    except OSError:
        return []


def _normalized(s):
    """去掉空白与标点，只留字母数字汉字，用于宽松比较（如「是，大臣 第一季」对「是,大臣第一季」）。"""
    return re.sub(r"[\s，,。.、·:：!！?？\-_（）()\[\]【】/\\]+", "", str(s)).lower()


def find_poster_file(title):
    """在 data/posters 下按文件名查找海报。

    匹配顺序：精确文件名、忽略大小写、去掉标点后相等（容忍「，」与「,」差异）。

    不做包含/前缀匹配：库里同时存在《功夫》和《功夫熊猫》这类互为前缀的片名时，
    模糊匹配会把另一部电影的海报显示给用户，比返回 None 更糟。
    缺图应由 scripts/fetch_posters.py 补齐。
    """
    title = to_str(title)
    if not title or not POSTER_DIR.is_dir():
        return None

    safe = _safe_name(title)

    exact = POSTER_DIR / f"{safe}.jpg"
    if exact.is_file():
        return exact

    files = [p for p in _poster_files() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    if not files:
        return None

    low = safe.lower()
    for p in files:  # 忽略大小写
        if p.stem.lower() == low:
            return p

    norm = _normalized(safe)
    if norm:
        for p in files:  # 去标点后相等
            if _normalized(p.stem) == norm:
                return p
    return None


def find_chart_file(name):
    mapping = {
        "boxplot": ANALYSIS_DIR / "genre_rating_box.png",
    }
    path = mapping.get(name)
    if path is not None and path.is_file():
        return path
    return None


# HTTP 处理
class ApiHandler(BaseHTTPRequestHandler):
    server_version = "MovieAgentHTTP/1.0"
    protocol_version = "HTTP/1.1"

    # 基础输出
    def _send_bytes(self, status, body, content_type, extra_headers=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in CORS_HEADERS:
                self.send_header(key, value)
            for key, value in (extra_headers or []):
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 客户端提前断开，忽略

    def send_json(self, obj, status=200):
        self._send_bytes(status, dumps(obj), "application/json; charset=utf-8")

    def send_error_json(self, message, status=400):
        self.send_json({"error": str(message)}, status=status)

    def send_file(self, path, content_type=None):
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            self.send_error_json(f"读取文件失败: {exc_text(exc)}", 500)
            return
        if content_type is None:
            content_type = MIME_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")
        self._send_bytes(200, data, content_type, [("Cache-Control", "no-cache")])

    # 请求分发
    def do_GET(self):
        try:
            self._route_get()
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(f"服务器内部错误: {exc_text(exc)}", 500)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        try:
            self._route_post()
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(f"服务器内部错误: {exc_text(exc)}", 500)

    def do_OPTIONS(self):
        self._send_bytes(204, b"", "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    # GET 路由
    def _route_get(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)

        def q(name, default=""):
            values = query.get(name)
            return values[0] if values else default

        if path == "/api/status":
            self.send_json(APP.status())
        elif path == "/api/movies":
            self.api_movies(q("q"), q("genre"), q("sort") or "rating", q("limit") or "200")
        elif path == "/api/movie":
            self.api_movie(q("title"))
        elif path == "/api/analysis":
            self.api_analysis()
        elif path == "/api/chart/boxplot":
            self.api_chart("boxplot")
        elif path == "/api/cinemas":
            self.api_cinemas(q("city") or "成都")
        elif path == "/api/poster":
            self.api_poster(q("title"))
        elif path.startswith("/api/"):
            self.send_error_json(f"未知接口: {path}", 404)
        else:
            self.serve_static(path)

    # POST 路由
    def _route_post(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        body = self._read_json_body()

        if path == "/api/chat":
            self.api_chat(body if isinstance(body, dict) else {})
        elif path == "/api/chat/reset":
            self.api_chat_reset()
        elif path.startswith("/api/"):
            self.send_error_json(f"未知接口: {path}", 404)
        else:
            self.send_error_json(f"不支持的请求: {path}", 404)

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    # 接口实现
    def api_movies(self, keyword, genre, sort, limit):
        APP.require_data()
        df = APP.movies_df

        limit = to_int(limit)
        if limit is None or limit <= 0:
            limit = 200
        limit = min(limit, 1000)

        mask = None
        keyword = to_str(keyword)
        if keyword:
            kw = keyword.lower()
            # 对 title / director / actors 做不区分大小写的模糊匹配
            cond = (
                df["title"].astype(str).str.lower().str.contains(kw, na=False, regex=False)
                | df["director"].astype(str).str.lower().str.contains(kw, na=False, regex=False)
                | df["actors"].astype(str).str.lower().str.contains(kw, na=False, regex=False)
            )
            mask = cond

        genre = to_str(genre)
        if genre:
            gcond = df["genre"].astype(str).str.contains(genre, na=False, case=False, regex=False)
            mask = gcond if mask is None else (mask & gcond)

        sub = df if mask is None else df[mask]

        sort = (sort or "rating").strip().lower()
        if sort == "year":
            sub = sub.sort_values("year", ascending=False, na_position="last")
        elif sort == "title":
            sub = sub.sort_values("title", ascending=True, na_position="last", key=lambda s: s.astype(str))
        else:
            sub = sub.sort_values("douban_rating", ascending=False, na_position="last")

        total = int(len(sub))
        items = [APP.movie_item(row) for _, row in sub.head(limit).iterrows()]
        self.send_json({"total": total, "items": items})

    def api_movie(self, title):
        APP.require_data()
        title = to_str(title)
        if not title:
            self.send_error_json("缺少参数 title", 400)
            return

        row = APP.find_movie_row(title)
        if row is None:
            self.send_json({"error": "not found"}, 404)
            return

        movie = APP.movie_item(row)

        rdf = APP.reviews_df
        good, bad = [], []
        if rdf is not None and len(rdf) > 0 and "title" in rdf.columns:
            sub = rdf[rdf["title"].astype(str).str.contains(title, na=False, case=False, regex=False)]
            sub = sub.dropna(subset=["rating"])
            if len(sub) > 0:
                sub = sub.copy()
                sub["rating"] = sub["rating"].astype(float)
                for _, r in sub.sort_values("rating", ascending=False).head(5).iterrows():
                    good.append(self._review_item(r))
                for _, r in sub.sort_values("rating", ascending=True).head(5).iterrows():
                    bad.append(self._review_item(r))

        similar = self._similar_movies(row)
        self.send_json({"movie": movie, "reviews": {"good": good, "bad": bad}, "similar": similar})

    @staticmethod
    def _review_item(r):
        return {
            "username": to_str(r.get("username")),
            "rating": to_float(r.get("rating")),
            "content": to_str(r.get("content")),
            "date": to_str(r.get("date")),
            "votes": to_int(r.get("votes")) or 0,
        }

    def _similar_movies(self, row, max_n=6):
        import pandas as pd

        df = APP.movies_df
        title = to_str(row.get("title"))
        genre_set = set(split_list(row.get("genre")))
        director = to_str(row.get("director"))
        director_ok = bool(director) and director not in ("未知", "nan")

        candidates = df[df["title"].astype(str) != title]
        if len(candidates) == 0:
            return []

        def score(r):
            hits = 0
            r_director = to_str(r.get("director"))
            if director_ok and r_director and r_director not in ("未知", "nan"):
                if r_director == director or r_director in director or director in r_director:
                    hits += 1
            if genre_set & set(split_list(r.get("genre"))):
                hits += 1
            return hits

        scored = [(score(r), r) for _, r in candidates.iterrows()]
        picked = [r for s, r in scored if s > 0]

        if not picked:
            # 没有同导演/同类型的，退回豆瓣评分最高的若干部
            picked = [r for _, r in candidates.iterrows()]

        picked.sort(key=lambda r: (to_float(r.get("douban_rating")) or -1.0), reverse=True)
        picked = picked[:max_n]

        out = []
        for r in picked:
            t = to_str(r.get("title"))
            out.append(
                {
                    "title": t,
                    "year": to_int(r.get("year")),
                    "genre": to_str(r.get("genre")),
                    "douban_rating": to_float(r.get("douban_rating")),
                    "poster": APP.poster_url(t),
                }
            )
        return out

    def api_chat(self, body):
        message = to_str(body.get("message"))
        reset = bool(body.get("reset"))
        if not message:
            self.send_error_json("message 不能为空", 400)
            return

        if reset:
            self.api_chat_reset(silent=True)

        with APP.init_lock:
            ready = APP.ready
            error = APP.error
        if not ready:
            hint = "模型正在初始化，请稍候" if not error else f"模型初始化失败: {error}"
            self.send_json({"error": hint}, 503)
            return

        started = time.time()
        result = self._call_agent(message)
        elapsed = int((time.time() - started) * 1000)

        if isinstance(result, dict) and result.get("__error__"):
            self.send_json({"error": result["__error__"]}, 503)
            return

        answer = to_str(result.get("answer")) if isinstance(result, dict) else to_str(result)
        poster_raw = result.get("poster_path") if isinstance(result, dict) else None
        trailer_raw = result.get("trailer_url") if isinstance(result, dict) else None

        movie_title = APP.extract_movie_title(message, answer)

        poster = self._public_poster(poster_raw)
        if poster is None and movie_title:
            poster = APP.poster_url(movie_title)

        # 海报确实缺失（本地缓存里没有这张图）时明确告知，方便前端区分「没数据」和「出错了」
        poster_missing = bool(movie_title) and poster is None

        trailer = to_str(trailer_raw) if not is_blank(trailer_raw) else None
        if trailer and not trailer.startswith("http"):
            trailer = None

        self.send_json(
            {
                "answer": answer,
                "poster": poster,
                "poster_missing": poster_missing,
                "trailer": trailer,
                "movie": movie_title,
                "elapsed_ms": elapsed,
            }
        )

    def _call_agent(self, message):
        """在锁里串行调用共享 agent（含超时保护）。"""
        box = {}

        def worker():
            try:
                with APP.lock:
                    box["result"] = APP.agent.chat_with_details(message)
            except Exception as exc:
                traceback.print_exc()
                box["error"] = exc_text(exc)

        thread = threading.Thread(target=worker, name="movie-agent-chat", daemon=True)
        thread.start()
        thread.join(CHAT_TIMEOUT_SEC)

        if thread.is_alive():
            return {"__error__": f"模型响应超时（>{int(CHAT_TIMEOUT_SEC)}秒），请稍后重试"}
        if "error" in box:
            return {"__error__": box["error"]}
        return box.get("result") or {"answer": "", "poster_path": None, "trailer_url": None}

    @staticmethod
    def _public_poster(raw):
        """本地海报路径 -> /api/poster?title=...；http(s) 原样返回。"""
        value = to_str(raw)
        if not value:
            return None
        low = value.lower()
        if low.startswith("http://") or low.startswith("https://"):
            return value
        try:
            path = Path(value)
        except (OSError, ValueError):
            return None
        if path.is_file():
            title = path.stem
            file_title = _safe_name(title)
            # 反查真实标题（文件名可能把非法字符替换成了 _）
            for t, _low in APP.title_cache:
                if _safe_name(t).lower() == file_title.lower():
                    title = t
                    break
            return f"/api/poster?title={quote(title, safe='')}"
        return None

    def api_chat_reset(self, silent=False):
        with APP.init_lock:
            ready = APP.ready
            error = APP.error
        if not ready or APP.agent is None:
            if silent:
                return
            hint = "模型正在初始化，请稍候" if not error else f"模型初始化失败: {error}"
            self.send_json({"error": hint}, 503)
            return
        with APP.lock:
            APP.agent.reset_history()
        if not silent:
            self.send_json({"ok": True})

    def api_analysis(self):
        if APP.cfg is None and not APP.load_config_once():
            self.send_error_json("配置加载失败", 500)
            return
        if APP.cfg is None:
            self.send_error_json("配置加载失败", 500)
            return
        try:
            self.send_json(APP.get_analysis())
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(f"分析失败: {exc_text(exc)}", 500)

    def api_chart(self, name):
        path = find_chart_file(name)
        if path is None:
            self.send_error_json(f"图表不存在: {name}", 404)
            return
        self.send_file(path, "image/png")

    def api_cinemas(self, city):
        city = to_str(city) or "成都"
        if APP.cfg is None:
            APP.load_config_once()
        cfg = APP.cfg or {}
        amap_key = ""
        try:
            amap_key = str((cfg.get("amap") or {}).get("api_key") or "")
        except Exception:
            amap_key = ""
        if not amap_key or amap_key == "YOUR_AMAP_KEY_HERE":
            amap_key = os.environ.get("AMAP_API_KEY", "").strip()
        if not amap_key or amap_key == "YOUR_AMAP_KEY_HERE":
            self.send_json({"error": "未配置高德地图 API Key"}, 200)
            return

        try:
            from src.cinema import search_nearby_cinemas

            raw = search_nearby_cinemas(amap_key, city)
        except Exception as exc:
            self.send_json({"error": f"高德地图查询失败: {exc_text(exc)}"}, 200)
            return

        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and raw[0].get("error"):
            self.send_json({"error": to_str(raw[0]["error"])}, 200)
            return

        items = []
        for c in raw or []:
            if not isinstance(c, dict):
                continue
            items.append(
                {
                    "name": to_str(c.get("name")),
                    "address": to_str(c.get("address")),
                    "location": to_str(c.get("location")),
                    "distance": to_str(c.get("distance")),
                    "tel": to_str(c.get("tel")),
                    "rating": to_str(c.get("rating")),
                    "map_url": to_str(c.get("map_url")),
                    "nav_url": to_str(c.get("nav_url")),
                    "ticket_url": to_str(c.get("ticket_url")),
                }
            )
        self.send_json({"city": city, "items": items})

    def api_poster(self, title):
        title = to_str(title)
        if not title:
            self.send_error_json("缺少参数 title", 400)
            return
        path = find_poster_file(title)
        if path is None:
            self.send_error_json(f"海报不存在: {title}", 404)
            return
        content_type = MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
        if content_type not in ("image/jpeg", "image/png", "image/webp"):
            content_type = "image/jpeg"
        self.send_file(path, content_type)

    # 静态文件
    def serve_static(self, path):
        path = path or "/"
        if path.endswith("/"):
            path = path + "index.html"

        rel = path.lstrip("/")
        if not rel:
            rel = "index.html"

        # 防路径穿越：统一分隔符 + 拒绝 ..
        segments = [s for s in re.split(r"[\\/]+", rel) if s and s != "."]
        if any(s == ".." for s in segments) or "\x00" in rel:
            self.send_error_json("非法路径", 400)
            return

        target = (WEB_DIR / Path(*segments)).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            self.send_error_json("非法路径", 403)
            return

        if not target.is_file():
            if not WEB_DIR.is_dir():
                self.send_error_json(
                    "web/index.html 尚未创建（前端还在搭建中），静态目录不存在: web/", 404
                )
                return
            self.send_error_json(f"文件不存在: /{rel}", 404)
            return

        self.send_file(target)


class MovieAgentServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def main():
    APP.start_init()

    server = MovieAgentServer((HOST, PORT), ApiHandler)
    print("=" * 62, flush=True)
    print("  🎬 电影查询智能体 HTTP API 服务", flush=True)
    print(f"  地址: http://{HOST}:{PORT}", flush=True)
    print(f"  根目录: {ROOT_DIR}", flush=True)
    print(f"  静态目录: {WEB_DIR}", flush=True)
    print("  模型/知识库正在后台初始化，可通过 GET /api/status 查看进度", flush=True)
    print("=" * 62, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 收到中断，正在关闭 ...", flush=True)
    finally:
        server.server_close()
        print("[server] 已关闭", flush=True)


if __name__ == "__main__":
    main()
