# -*- coding: utf-8 -*-
"""
简易电影信息爬取模块
核心策略：用无反爬的公开接口快速获取海报/基本信息
- 豆瓣 suggest 接口：公开 JSON，无鉴权，直接返回海报URL
- 图片下载：GUI端用 QtNetwork 带 Referer 加载
"""
import re
import time
import requests
from src.utils import logger, ROOT_DIR, ensure_dir


class MovieCrawler:
    DOUBAN_SUGGEST = "https://movie.douban.com/j/subject_suggest"
    TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
    TMDB_MOVIE = "https://api.themoviedb.org/3/movie/{}"
    TMDB_VIDEOS = "https://api.themoviedb.org/3/movie/{}/videos"

    def __init__(self, cfg):
        self.cfg = cfg
        self.tmdb_key = cfg["crawler"].get("tmdb_api_key", "")
        self.delay = cfg["crawler"]["request_delay"]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://movie.douban.com/",
        }
        self.posters_dir = ROOT_DIR / cfg["data"]["posters_dir"]
        ensure_dir(self.posters_dir)

    def search_movie(self, title):
        """搜索电影，优先TMDB，其次豆瓣suggest（无反爬）"""
        if self.tmdb_key:
            info = self._search_tmdb(title)
            if info:
                return info
        return self._search_douban(title)

    def download_poster(self, url, title, try_variants=True):
        """下载海报到本地（带Referer头防403 + 严格图片校验）"""
        if not url:
            return None
        safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
        save_path = self.posters_dir / f"{safe_title}.jpg"
        if save_path.exists() and save_path.stat().st_size > 2000:
            # 已有文件且大小正常（>2KB 基本不是 HTML 错页）
            ok = False
            try:
                from PIL import Image
                with Image.open(save_path) as im:
                    im.verify()
                    w, h = im.size
                ok = True
            except Exception:
                save_path.unlink(missing_ok=True)   # 文件损坏，重新下载
            if ok:
                # 豆瓣默认给 s_ratio_poster（270x377）小图，若缓存里只有小图，
                # 这里直接返回会导致大图永远补不上，所以低于 200px 时重新下载
                if try_variants and (w < 200 or h < 200):
                    save_path.unlink(missing_ok=True)
                    logger.info(f"已有海报过小({w}x{h})，重新下载大图: {title}")
                else:
                    return str(save_path)

        # 候选 URL：大图优先，原始地址兜底。豆瓣尺寸有 s/m/l_ratio_poster，
        # CDN 有 img1/3/9 等子域名；若把 s_ 小图排在前面会直接命中缩略图。
        urls_to_try = [url]
        if try_variants:
            variants = []
            for size in ("l_ratio_poster", "m_ratio_poster", "l_poster", "m_poster", "s_poster"):
                for cdn in ("img1", "img2", "img3", "img9"):
                    v = re.sub(r'img\d\.doubanio\.com/view/photo/[^/]+/', f'{cdn}.doubanio.com/view/photo/{size}/', url)
                    if v != url and v not in variants:
                        variants.append(v)
            urls_to_try = variants + [url]

        for try_url in urls_to_try:
            try:
                r = requests.get(try_url, headers=self.headers, timeout=15, allow_redirects=True)
                r.raise_for_status()
                content = r.content
                ct = r.headers.get("Content-Type", "").lower()
                is_image = "image" in ct or content[:3] in (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"RIFF")
                if not is_image or len(content) < 1000:
                    logger.warning(f"海报非图片或太小({len(content)}B): {try_url} ct={ct}")
                    continue
                try:
                    from io import BytesIO
                    from PIL import Image
                    im = Image.open(BytesIO(content))
                    im.verify()
                    # verify() 之后指针已失效，需要重新打开
                    im = Image.open(BytesIO(content))
                    im.load()
                    w, h = im.size
                    if w < 50 or h < 50:
                        logger.warning(f"海报尺寸太小 {w}x{h}: {try_url}")
                        continue
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    im.save(save_path, "JPEG", quality=88)
                    logger.info(f"海报已下载({w}x{h}): {save_path} <- {try_url}")
                    return str(save_path)
                except Exception as e:
                    logger.warning(f"海报PIL校验失败: {try_url} err={e}")
                    continue
            except Exception as e:
                logger.warning(f"海报下载失败 ({title}) {try_url}: {e}")
                continue
        logger.warning(f"海报所有变体均失败: {title}")
        return None

    def _search_tmdb(self, title):
        try:
            r = requests.get(self.TMDB_SEARCH, params={
                "api_key": self.tmdb_key, "query": title, "language": "zh-CN"
            }, headers=self.headers, timeout=10)
            data = r.json()
            results = data.get("results", [])
            if not results:
                return None
            m = results[0]
            mid = m["id"]
            time.sleep(self.delay)
            rd = requests.get(self.TMDB_MOVIE.format(mid), params={
                "api_key": self.tmdb_key, "language": "zh-CN"
            }, headers=self.headers, timeout=10).json()
            poster_url = f"https://image.tmdb.org/t/p/w500{rd.get('poster_path','')}" if rd.get("poster_path") else ""
            trailer_url = self._trailer_tmdb(mid)
            return {
                "source": "tmdb", "tmdb_id": mid,
                "title": rd.get("title") or title,
                "year": rd.get("release_date", "")[:4],
                "director": "", "actors": "",
                "genre": "/".join([g["name"] for g in rd.get("genres", [])]),
                "duration": rd.get("runtime"),
                "region": rd.get("origin_country", [""])[0] if rd.get("origin_country") else "",
                "summary": rd.get("overview", ""),
                "douban_rating": rd.get("vote_average"),
                "poster_url": poster_url, "trailer_url": trailer_url,
            }
        except Exception as e:
            logger.warning(f"TMDB搜索失败: {e}")
            return None

    def _bilibili_search_url(self, title):
        """生成B站预告片搜索链接（国内可直接打开）"""
        from urllib.parse import quote
        return f"https://search.bilibili.com/all?keyword={quote(title + '+预告片')}"

    def _trailer_tmdb(self, mid):
        """TMDB只用来拿中文名，最终返回B站搜索链接（国内可直接看）"""
        try:
            rd = requests.get(self.TMDB_MOVIE.format(mid), params={
                "api_key": self.tmdb_key, "language": "zh-CN"
            }, headers=self.headers, timeout=10).json()
            title = rd.get("title") or ""
            if title:
                return self._bilibili_search_url(title)
        except Exception:
            pass
        return ""

    def _search_douban(self, title):
        """只用豆瓣suggest接口（完全公开，无反爬），返回基本信息+海报URL"""
        try:
            r = requests.get(self.DOUBAN_SUGGEST, params={"q": title},
                             headers=self.headers, timeout=10)
            results = r.json()
            if not results:
                return None
            # 优先找movie类型
            movie_item = None
            for item in results:
                if item.get("type") == "movie":
                    movie_item = item
                    break
            if not movie_item:
                movie_item = results[0]

            info = {
                "source": "douban",
                "douban_id": movie_item.get("id"),
                "title": movie_item.get("title", title),
                "year": movie_item.get("year", ""),
                "poster_url": movie_item.get("img", ""),
                "summary": "", "director": "", "actors": "", "genre": "",
                "duration": None, "region": "", "douban_rating": None, "trailer_url": "",
            }
            # 从 sub_title 解析导演/演员（格式："导演 / 演员1 / 演员2 / 类型"）
            sub_title = movie_item.get("sub_title", "")
            if sub_title:
                parts = [p.strip() for p in sub_title.split("/")]
                if parts:
                    info["director"] = parts[0]
                if len(parts) > 1:
                    info["actors"] = "/".join(parts[1:min(4, len(parts))])
            return info
        except Exception as e:
            logger.warning(f"豆瓣搜索失败: {e}")
            return None
