# -*- coding: utf-8 -*-
"""Agent 工具实现：片单筛选与影片查询。"""
import os
import re

import pandas as pd

from src.utils import logger, ROOT_DIR, rating_label


class MovieTools:
    """供 MovieAgent 注册为工具的方法集合。"""

    def __init__(self, cfg, kb, crawler=None, online=None):
        self.cfg = cfg
        self.kb = kb
        self.crawler = crawler
        self.online = online
        self._df = None

    @property
    def df(self):
        """本地电影表，延迟加载一次。"""
        if self._df is None:
            path = ROOT_DIR / self.cfg["data"]["movies_csv"]
            if path.exists():
                self._df = pd.read_csv(path, encoding="utf-8-sig")
            else:
                self._df = pd.DataFrame()
        return self._df

    def local_titles(self):
        if self.df.empty or "title" not in self.df.columns:
            return []
        return [str(t).strip() for t in self.df["title"].tolist() if str(t).strip()]

    def local_year_range(self):
        if self.df.empty or "year" not in self.df.columns:
            return (None, None)
        years = pd.to_numeric(self.df["year"], errors="coerce").dropna()
        if years.empty:
            return (None, None)
        return (int(years.min()), int(years.max()))

    def has_local(self, name):
        """本地是否存在该片名（包含匹配，与 kb.get_movie_info 行为一致）。"""
        name = str(name or "").strip()
        if not name or self.df.empty:
            return False
        try:
            return bool(self.df["title"].astype(str).str.contains(name, na=False, case=False, regex=False).any())
        except Exception:
            return False

    def list_movies(self, year: str = "", genre: str = "", min_rating: float = 0, limit: int = 15) -> str:
        """按年份、类型、评分筛选本地影片。

        year: "2025" 或 "2020-2025" 区间，留空不限
        genre: 类型关键词，如 "科幻"，留空不限
        min_rating: 最低豆瓣评分，0 表示不限
        limit: 返回条数上限
        """
        if self.df.empty:
            return "本地电影数据为空，请先运行数据清洗（python main.py --clean）。"

        df = self.df.copy()
        df["_year"] = pd.to_numeric(df.get("year"), errors="coerce")
        df["_rating"] = pd.to_numeric(df.get("douban_rating"), errors="coerce")
        cond = pd.Series(True, index=df.index)

        # 年份：支持单年与区间
        y_min = y_max = None
        year = str(year or "").strip()
        if year:
            m = re.match(r"^\s*(\d{4})\s*[-~到至]\s*(\d{4})\s*$", year)
            if m:
                y_min, y_max = int(m.group(1)), int(m.group(2))
            elif re.match(r"^\d{4}$", year):
                y_min = y_max = int(year)
            else:
                m2 = re.search(r"(\d{4})", year)
                if m2:
                    y_min = y_max = int(m2.group(1))
        if y_min is not None:
            cond &= df["_year"].between(y_min, y_max, inclusive="both")

        # 类型
        genre = str(genre or "").strip()
        if genre:
            cond &= df.get("genre", pd.Series("", index=df.index)).astype(str).str.contains(
                re.escape(genre), na=False, case=False)

        # 评分
        try:
            min_rating = float(min_rating or 0)
        except (TypeError, ValueError):
            min_rating = 0.0
        if min_rating > 0:
            cond &= df["_rating"] >= min_rating

        hit = df[cond].sort_values("_rating", ascending=False)
        if hit.empty:
            lo, hi = self.local_year_range()
            span = f"{lo}-{hi}" if lo else "未知"
            return (f"本地知识库中没有符合条件的影片（当前收录 {len(df)} 部，年份范围 {span}）。\n"
                    f"提示：如果用户问的是更新年份或更冷门的电影，可改用 search_movie_web 联网查询，"
                    f"并明确告知「本地未收录」。")

        try:
            limit = max(1, min(int(limit or 15), 50))
        except (TypeError, ValueError):
            limit = 15

        lines = [f"📚 本地知识库筛选结果（共命中 {len(hit)} 部，显示前 {min(limit, len(hit))} 部）"]
        for _, row in hit.head(limit).iterrows():
            rating = row.get("_rating")
            rating_s = f"{rating:.1f}" if pd.notna(rating) else "暂无"
            year_s = f"{int(row['_year'])}" if pd.notna(row.get("_year")) else "—"
            lines.append(f"  · 《{row.get('title','')}》({year_s}) — {row.get('genre','')}，"
                         f"豆瓣 {rating_s}，导演 {row.get('director','未知')}")
        lines.append("\n如需某部影片的详细信息、评论、海报，可继续调用对应工具。")
        return "\n".join(lines)

    def search_movie_web(self, movie_name: str) -> str:
        """查询影片信息，先查本地知识库，未收录再联网抓取。

        movie_name: 电影名称
        """
        name = str(movie_name or "").strip()
        if not name:
            return "请提供电影名称。"

        info = None
        try:
            info = self.kb.get_movie_info(name)
        except Exception as exc:
            logger.warning(f"本地知识库查询失败: {exc}")

        if info:
            dr = info.get("douban_rating")
            mr = info.get("maoyan_rating")
            lines = [f"📚 来自本地知识库（未联网）",
                     f"🎬 《{info.get('title','')}》({info.get('year','')})"]
            if info.get("director"):
                lines.append(f"🎥 导演: {info['director']}")
            if info.get("actors"):
                lines.append(f"⭐ 主演: {info['actors']}")
            if info.get("genre"):
                lines.append(f"🎭 类型: {info['genre']}")
            if info.get("duration"):
                lines.append(f"⏱️ 片长: {info['duration']}分钟")
            if info.get("region"):
                lines.append(f"🌍 地区: {info['region']}")
            lines.append(f"📊 豆瓣: {dr if pd.notna(dr) else '暂无'}  猫眼: {mr if pd.notna(mr) else '暂无'}")
            if pd.notna(dr):
                lines.append(f"   评级: {rating_label(float(dr))}")
            if info.get("summary"):
                lines.append(f"\n📝 简介: {info['summary']}")
            return "\n".join(lines)

        if self.online is None:
            return f"本地知识库没有《{name}》，且联网数据源不可用。"

        logger.info(f"本地未收录《{name}》，转联网抓取豆瓣")
        detail, candidates = None, []
        try:
            detail, candidates = self.online.search_detail(name)
        except Exception as exc:
            logger.warning(f"联网抓取异常: {exc}")

        if detail:
            return self.online.format_for_agent(detail, candidates)

        err = getattr(self.online, "last_error", "") or ""
        lo, hi = self.local_year_range()
        tip = f"本地知识库当前收录 {len(self.local_titles())} 部影片（年份 {lo}-{hi}）。" if lo else ""
        return (f"未能获取《{name}》的信息。\n{tip}\n"
                + (f"联网抓取失败原因：{err}\n" if err else "")
                + "请稍后重试，或换一个更准确的片名。注意：不要编造评分或评论。")
