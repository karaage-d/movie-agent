# -*- coding: utf-8 -*-
"""
联网实时数据源（豆瓣）

为什么单独一个模块：
  项目原本只有 crawler.py，它靠「豆瓣 suggest 接口」拿片名/年份/海报，**没有评分**。
  而豆瓣桌面详情页（movie.douban.com/subject/<id>/）会被反爬重定向到 sec.douban.com，
  换 UA 也没用。实测可用的是**手机站 m.douban.com**（必须用手机 UA），页面里有
  itemprop="ratingValue" / reviewCount 以及 sub-meta 元信息，因此本模块专门走手机站。

实测可用性（2026-10 于本机）：
  ✅ movie.douban.com/j/subject_suggest   搜索：片名/年份/海报/ID
  ✅ m.douban.com/movie/subject/<id>/     详情：评分/评价数/类型/地区/片长/上映/简介/海报
  ❌ movie.douban.com/subject/<id>/       被反爬拦截（302 -> sec.douban.com）
  ❌ api.themoviedb.org / 猫眼 / 百度百科  网络不通或 403

数据来源都会标注为「豆瓣(手机版)」，供 Agent 在回复里标注【📡 网络实时数据】。
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
    """豆瓣实时数据源（搜索 + 详情解析）。

    重要：豆瓣对高频请求会返回**空数组/空页**（HTTP 200）作为软限流，
    因此这里做了三层保护：
      1. 内存缓存（同一部片只抓一次）；
      2. 请求之间强制 sleep（delay，默认 3 秒，可在 config 调整）；
      3. 命中限流时按 backoff 递增等待重试，仍失败则明确抛出可读错误，
         由 Agent 如实告知用户，而不是编造数据。
    """

    def __init__(self, cfg=None, delay=3.0, backoff=(6, 18)):
        self.cfg = cfg or {}
        self.delay = delay
        # 限流重试间隔：一次命中限流最多多等 24 秒。设太长会让对话卡到超时（服务端 90s 上限），
        # 所以宁可少试几次、快速失败并把「被限流」如实告诉用户。
        self.backoff = backoff
        self._cache = {}          # sid -> 详情
        self._suggest_cache = {}  # 关键词 -> 候选列表
        self.last_error = ""
        self._last_request_at = 0.0

    def _throttle(self):
        """保证两次请求之间至少间隔 delay 秒。"""
        gap = time.time() - self._last_request_at
        if self._last_request_at and gap < self.delay:
            time.sleep(self.delay - gap)
        self._last_request_at = time.time()

    def _get(self, url, params=None, mobile=False, timeout=15):
        """带节流 + 限流重试的 GET。返回 Response 或 None。"""
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

    # ---------------- 搜索 ----------------
    def suggest(self, keyword):
        """按关键词搜索，返回候选列表（每项含 id/title/year/poster_url）。"""
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
        # 空数组是豆瓣的软限流信号
        if not isinstance(data, list) or not data:
            self.last_error = ("豆瓣返回空结果（通常是请求过快被限流）。"
                               "请等 1-2 分钟再试，或减少连续查询次数。")
            logger.warning(f"豆瓣 suggest 空结果: {keyword}")
            return []

        picked = data
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
                "rating": None,        # 需要抓详情页才有
            })
        if out:
            self._suggest_cache[keyword] = out
        return out

    # ---------------- 详情 ----------------
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
        """搜索并抓第一条的详情（最常用的入口）。返回 (detail, candidates)。"""
        candidates = self.suggest(keyword)
        if not candidates:
            return None, []

        detail = self.fetch_detail(candidates[0]["douban_id"])
        if detail is None:
            # 详情拿不到，至少把搜索到的概要返回
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

    # ---------------- 解析 ----------------
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
        #           或：美国 / 英国 / 剧情 / 传记 / 历史 / 2023-08-30(中国大陆)上映 / 片长180分钟
        parts = [p.strip() for p in re.split(r"[/／]", meta or "") if p.strip()]
        region_parts, genres, release, duration = [], [], "", ""
        for p in parts:
            if "上映" in p:
                # 「2023-08-30(中国大陆)上映」-> 去掉「上映」和括号里的地区。
                # 豆瓣页面的括号中英文混用（见过「2025-12-19（中国大陆)」），所以两种括号都清，
                # 并额外兜一层：万一括号本身缺失，就把尾部的「XX地区)」样式残留也去掉。
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
                # 地区出现在类型之前；「中国大陆」这类不以「国」结尾的靠"类型词尚未出现"来兜
                region_parts.append(p)
            else:
                genres.append(p)

        # 简介：优先 meta description（形如「奥本海默豆瓣评分：8.8 简介：……」）
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

    # ---------------- 供 Agent 使用的文本格式化 ----------------
    @staticmethod
    def format_for_agent(info, candidates=None):
        if not info:
            return "未在豆瓣上找到相关影片信息。"
        lines = [f"📡 网络实时数据（来源：豆瓣手机版）"]
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
