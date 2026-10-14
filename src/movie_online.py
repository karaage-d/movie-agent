# -*- coding: utf-8 -*-
"""豆瓣实时数据源：搜索影片并解析详情。

详情走手机站 m.douban.com（需手机 UA），桌面详情页 movie.douban.com/subject/<id>/
有反爬，会 302 到 sec.douban.com。
"""
import json
import re
import time
import urllib.parse

import requests

from src.utils import logger

UA_DESKTOP = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
UA_MOBILE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")

SUGGEST_URL = "https://movie.douban.com/j/subject_suggest"
SUBJECT_URL = "https://m.douban.com/movie/subject/{sid}/"

# 常见类型词，用于把 sub-meta 里的「国家/地区」和「类型」区分开
_GENRE_WORDS = {
    "剧情", "喜剧", "动作", "爱情", "科幻", "动画", "悬疑", "犯罪", "纪录片", "恐怖",
    "战争", "传记", "历史", "冒险", "奇幻", "惊悚", "灾难", "武侠", "古装", "音乐",
    "歌舞", "家庭", "儿童", "运动", "短片", "情色", "同性", "西部", "黑色电影",
    "脱口秀", "真人秀", "戏曲", "鬼怪", "荒诞",
}

# 地区词常见结尾，用于把「中国大陆 / 美国 / 香港」和类型词区分开
_REGION_SUFFIX = ("国", "大陆", "香港", "台湾", "地区", "英国", "法国", "日本", "韩国", "印度")


class MovieOnlineSource:
    """豆瓣搜索与详情解析。请求带节流与重试，避免触发限流。"""

    def __init__(self, cfg=None, delay=3.0, backoff=(6, 18)):
        self.cfg = cfg or {}
        self.delay = delay
        # 限流后的重试间隔。等太久会撞上服务端 90 秒的对话超时，所以快速失败更好。
        self.backoff = backoff
        self._cache = {}          # sid -> 详情
        self._suggest_cache = {}  # 关键词 -> 候选列表
        self.last_error = ""
        self._last_request_at = 0.0

    def _throttle(self):
        """控制请求间隔，至少 delay 秒。"""
        gap = time.time() - self._last_request_at
        if self._last_request_at and gap < self.delay:
            time.sleep(self.delay - gap)
        self._last_request_at = time.time()

    def _get(self, url, params=None, mobile=False, timeout=15):
        """GET 请求，带节流与限流重试。失败返回 None。"""
        headers = {"User-Agent": UA_MOBILE if mobile else UA_DESKTOP,
                   "Accept-Language": "zh-CN,zh;q=0.9",
                   "Referer": "https://m.douban.com/" if mobile else "https://movie.douban.com/"}
        last_exc = None
        for attempt, wait in enumerate((0,) + tuple(self.backoff)):
            if wait:
                time.sleep(wait)
            self._throttle()
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    return r
                last_exc = f"HTTP {r.status_code}"
            except Exception as exc:
                last_exc = exc
        self.last_error = f"请求豆瓣失败（已重试 {len(self.backoff)} 次）: {last_exc}"
        logger.warning(f"{self.last_error} <- {url}")
        return None

    def suggest(self, keyword):
        """按关键词搜索，返回候选列表。"""
        keyword = str(keyword or "").strip()
        if not keyword:
            return []
        if keyword in self._suggest_cache:
            return self._suggest_cache[keyword]

        r = self._get(SUGGEST_URL, params={"q": keyword})
        if r is None:
            return []
        try:
            data = r.json()
        except Exception:
            data = None
        # 豆瓣限流时会返回空数组
        if not isinstance(data, list) or not data:
            self.last_error = ("豆瓣返回空结果（通常是请求过快被限流）。"
                               "请等 1-2 分钟再试，或减少连续查询次数。")
            logger.warning(f"豆瓣 suggest 空结果: {keyword}")
            return []

        movies = [it for it in data if isinstance(it, dict) and it.get("type") == "movie"]
        picked = (movies or [it for it in data if isinstance(it, dict)])[:5]
        out = []
        for it in picked:
            sid = str(it.get("id") or "").strip()
            if not sid:
                continue
            out.append({
                "source": "douban",
                "douban_id": sid,
                "title": str(it.get("title") or "").strip(),
                "year": str(it.get("year") or "").strip(),
                "original_title": str(it.get("sub_title") or "").strip(),
                "poster_url": str(it.get("img") or "").strip(),
                "url": f"https://movie.douban.com/subject/{sid}/",
                "rating": None,        # 评分需另抓详情页
            })
        if out:
            self._suggest_cache[keyword] = out
        return out

    def fetch_detail(self, sid, use_cache=True):
        """抓 m.douban.com 详情页并解析。失败返回 None。"""
        sid = str(sid or "").strip()
        if not sid:
            return None
        if use_cache and sid in self._cache:
            return self._cache[sid]

        url = SUBJECT_URL.format(sid=sid)
        r = self._get(url, mobile=True)
        if r is None:
            return None
        html = r.text
        if len(html) < 5000 or "sec.douban.com" in r.url:
            self.last_error = "豆瓣详情页被反爬拦截（302 -> sec.douban.com）"
            logger.warning(f"{self.last_error}: {url}")
            return None

        info = self._parse(html, sid)
        if info:
            self._cache[sid] = info
        return info

    def search_detail(self, keyword):
        """搜索并抓取第一条结果的详情。返回 (detail, candidates)。"""
        candidates = self.suggest(keyword)
        if not candidates:
            return None, []

        detail = self.fetch_detail(candidates[0]["douban_id"])
        if detail is None:
            # 详情抓不到时退回搜索结果的概要信息
            c = candidates[0]
            return {
                "source": "douban-suggest",
                "title": c["title"], "year": c["year"],
                "original_title": c["original_title"],
                "poster_url": c["poster_url"], "url": c["url"],
                "rating": None, "votes": None, "genres": [],
                "region": "", "release": "", "duration": "",
                "summary": "", "partial": True,
            }, candidates
        return detail, candidates

    @staticmethod
    def _parse(html, sid):
        def grab(pattern, flags=0):
            m = re.search(pattern, html, flags)
            return m.group(1).strip() if m else ""

        title = grab(r'<div class="sub-title">(.*?)</div>')
        original = grab(r'<div class="sub-original-title">(.*?)</div>')
        rating = grab(r'itemprop="ratingValue"\s+content="([\d.]+)"')
        votes = grab(r'itemprop="reviewCount"\s+content="(\d+)"')
        meta = grab(r'<div class="sub-meta">(.*?)</div>', re.S)
        poster = grab(r'<a class="sub-cover"[^>]*>\s*<img src="([^"]+)"')
        desc = grab(r'<meta name="description" content="([^"]*)"')

        if not title:
            og = grab(r'<meta property="og:title" content="([^"]+)"')
            title = og.replace(" - 电影", "").strip()
        if not title:
            return None

        # sub-meta 形如：中国大陆 / 科幻 / 冒险 / 灾难 / 2023-01-22(中国大陆)上映 / 片长173分钟
        parts = [p.strip() for p in re.split(r"[/／]", meta or "") if p.strip()]
        region_parts, genres, release, duration = [], [], "", ""
        for p in parts:
            if "上映" in p:
                # 去掉「上映」及括号里的地区，豆瓣页面的括号中英文混用（如「（中国大陆)」），
                # 所以两种括号都清，另外再兜一层尾部残留。
                release = re.sub(r"[（(].*?[）)]", "", p.replace("上映", "")).strip()
                release = re.sub(r"[\u4e00-\u9fff]{2,4}[）)]$", "", release).strip(" )）")
            elif "片长" in p:
                duration = re.sub(r"[^\d]", "", p)
            elif re.match(r"^\d{4}", p):
                release = release or p
            elif p in _GENRE_WORDS:
                genres.append(p)
            elif p.endswith(_REGION_SUFFIX) or (not genres and len(region_parts) < 3
                                                and re.match(r"^[\u4e00-\u9fff]{2,4}$", p)):
                # 地区排在类型前面；「中国大陆」不以「国」结尾，靠类型词尚未出现来判断
                region_parts.append(p)
            else:
                genres.append(p)

        # 简介优先取 meta description，形如「奥本海默豆瓣评分：8.8 简介：……」
        summary = ""
        if desc:
            m = re.search(r"简介：(.*)$", desc, re.S)
            summary = (m.group(1) if m else desc).strip()

        year = ""
        m = re.search(r"（(\d{4})）", original)
        if m:
            year = m.group(1)
        if not year:
            m = re.search(r"(\d{4})", release)
            year = m.group(1) if m else ""

        return {
            "source": "douban",
            "douban_id": sid,
            "title": title,
            "original_title": original,
            "year": year,
            "rating": float(rating) if rating else None,
            "votes": int(votes) if votes else None,
            "genres": genres,
            "region": " / ".join(region_parts),
            "release": release,
            "duration": duration,
            "summary": summary,
            "poster_url": poster,
            "url": f"https://movie.douban.com/subject/{sid}/",
            "partial": False,
        }

    @staticmethod
    def format_for_agent(info, candidates=None):
        """把解析结果拼成给模型看的文本。"""
        if not info:
            return "未在豆瓣上找到相关影片信息。"
        lines = ["📡 网络实时数据（来源：豆瓣手机版）"]
        lines.append(f"🎬 《{info.get('title','')}》"
                     + (f"（{info['year']}）" if info.get("year") else ""))
        if info.get("original_title"):
            lines.append(f"🔤 原名: {info['original_title']}")
        r = info.get("rating")
        if r:
            votes = info.get("votes")
            lines.append(f"📊 豆瓣评分: {r}" + (f"（{votes} 人评价）" if votes else ""))
        else:
            lines.append("📊 豆瓣评分: 暂无（该片可能尚未开分）")
        if info.get("genres"):
            lines.append(f"🎭 类型: {' / '.join(info['genres'])}")
        if info.get("region"):
            lines.append(f"🌍 地区: {info['region']}")
        if info.get("release"):
            lines.append(f"📅 上映: {info['release']}")
        if info.get("duration"):
            lines.append(f"⏱️ 片长: {info['duration']}分钟")
        if info.get("summary"):
            lines.append(f"\n📝 简介: {info['summary'][:300]}")
        if info.get("poster_url"):
            lines.append(f"\n🖼️ 海报: {info['poster_url']}")
        if info.get("url"):
            lines.append(f"🔗 豆瓣页: {info['url']}")
        if len(candidates or []) > 1:
            others = "、".join(
                f"{c['title']}({c['year']})" for c in candidates[1:4] if c.get("title"))
            if others:
                lines.append(f"\nℹ️ 同名/相关候选: {others}")
        return "\n".join(lines)
