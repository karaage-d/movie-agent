# -*- coding: utf-8 -*-
"""Agent 工具实现：片单筛选与影片查询。"""
import os
import re

import pandas as pd

from src.utils import logger, ROOT_DIR, rating_label, is_blank


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

    def local_poster_path(self, title):
        """本地海报文件路径，没有则 None（只做精确/去标点匹配，不做模糊匹配）。"""
        import re

        title = str(title or "").strip()
        if not title:
            return None
        posters_dir = ROOT_DIR / self.cfg["data"]["posters_dir"]
        safe = re.sub(r'[\\/:*?"<>|]', "_", title)
        exact = posters_dir / f"{safe}.jpg"
        if exact.is_file():
            return str(exact)
        norm = lambda s: re.sub(r"[\s，,。.、·:：!！?？\-_（）()\[\]【】/\\]+", "", str(s)).lower()
        target = norm(safe)
        if not target or not posters_dir.is_dir():
            return None
        for p in sorted(posters_dir.iterdir()):
            if p.is_file() and norm(p.stem) == target:
                return str(p)
        return None

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

    def _merge_online(self, info, force=False):
        """联网补齐本地缺失的字段，本地已有的权威数据保持不变。

        本地库是权威来源（评分、猫眼分、评论、好评率都基于它），所以只补空字段：
        本地写了的绝不被联网覆盖。force=True 时用于「本地完全没收录」的情况。
        """
        if self.online is None:
            return None, ""
        title = str(info.get("title") or "").strip()
        if not title:
            return None, ""

        detail, candidates = None, []
        try:
            detail, candidates = self.online.search_detail(title)
        except Exception as exc:
            logger.warning(f"联网补充失败: {exc}")
            return None, ""
        if not detail:
            return None, getattr(self.online, "last_error", "") or ""

        filled = []
        if not force:
            pairs = [
                ("director", "导演"), ("actors", "主演"), ("genre", "类型"),
                ("region", "地区"), ("duration", "片长"), ("summary", "简介"),
                ("douban_rating", "豆瓣评分"),
            ]
            for key, label in pairs:
                if is_blank(info.get(key)) and not is_blank(detail.get(key)):
                    info[key] = detail[key]
                    filled.append(label)
            # 预告片本地普遍为空（实测 70/70 都空），联网拿到就用
            if is_blank(info.get("trailer_url")) and detail.get("url"):
                info["trailer_url"] = detail["url"]
                filled.append("豆瓣页")
        else:
            filled = ["全部"]
        return detail, ("、".join(filled) if filled else "")

    def search_movie_web(self, movie_name: str, need_trailer: str = "",
                         need_poster: str = "", force_refresh: str = "") -> str:
        """查一部电影：本地知识库优先，并用联网数据补齐本地缺失的字段。

        movie_name: 电影名称
        need_trailer: 传 "1" 表示特别需要预告片/豆瓣页链接
        need_poster: 传 "1" 表示特别需要海报
        force_refresh: 传 "1" 表示本地查不到也强制联网
        """
        name = str(movie_name or "").strip()
        if not name:
            return "请提供电影名称。"
        want_trailer = str(need_trailer).strip() in ("1", "true", "True", "是")
        want_poster = str(need_poster).strip() in ("1", "true", "True", "是")

        info = None
        try:
            info = self.kb.get_movie_info(name)
        except Exception as exc:
            logger.warning(f"本地知识库查询失败: {exc}")

        # 本地命中：以本地为主，联网只补空字段
        detail, filled = (None, "")
        if info:
            logger.info(f"本地命中《{name}》，联网补齐缺失字段")
            detail, filled = self._merge_online(info)

        # 本地没命中：完全靠联网
        if not info:
            if self.online is None:
                return f"本地知识库没有《{name}》，且联网数据源不可用。"
            logger.info(f"本地未收录《{name}》，转联网抓取豆瓣")
            try:
                detail, candidates = self.online.search_detail(name)
            except Exception as exc:
                logger.warning(f"联网抓取异常: {exc}")
                detail, candidates = None, []
            if not detail:
                err = getattr(self.online, "last_error", "") or ""
                lo, hi = self.local_year_range()
                tip = f"本地知识库当前收录 {len(self.local_titles())} 部影片（年份 {lo}-{hi}）。" if lo else ""
                return (f"未能获取《{name}》的信息。\n{tip}\n"
                        + (f"联网抓取失败原因：{err}\n" if err else "")
                        + "请稍后重试，或换一个更准确的片名。注意：不要编造评分或评论。")
            return self.online.format_for_agent(detail, candidates)

        # 组装：本地字段 + 标注哪些来自联网
        lines = []
        if filled:
            lines.append(f"📚 本地知识库（评分与评论以本地为准）+ 📡 联网补充：{filled}")
        else:
            lines.append("📚 来自本地知识库")
        lines.append(f"🎬 《{info.get('title','')}》({info.get('year','')})")
        for label, key in (("导演", "director"), ("主演", "actors"), ("类型", "genre"),
                           ("片长", "duration"), ("地区", "region")):
            v = info.get(key)
            if not is_blank(v):
                suffix = "分钟" if key == "duration" else ""
                lines.append(f"{label}: {v}{suffix}")

        dr, mr = info.get("douban_rating"), info.get("maoyan_rating")
        lines.append(f"📊 豆瓣: {dr if not is_blank(dr) else '暂无'}  猫眼: {mr if not is_blank(mr) else '暂无'}")
        if not is_blank(dr):
            try:
                lines.append(f"   评级: {rating_label(float(dr))}")
            except (TypeError, ValueError):
                pass
        if not is_blank(info.get("summary")):
            lines.append(f"\n📝 简介: {info['summary']}")

        # 预告片：本地没有就用联网拿到的豆瓣页
        trailer = info.get("trailer_url") or (detail or {}).get("url") or ""
        if trailer:
            lines.append(f"\n🎬 预告片/豆瓣页: {trailer}")
        elif want_trailer:
            lines.append("\n🎬 预告片: 暂未找到，可用 get_trailer 生成 B 站搜索链接")

        # 海报：本地有就给本地，没有则给联网图片地址
        local_poster = self.local_poster_path(info.get("title") or name)
        if local_poster:
            lines.append(f"\n🖼️ 海报: 本地 {local_poster}")
        elif (detail or {}).get("poster_url"):
            lines.append(f"\n🖼️ 海报(联网): {detail['poster_url']}")
        elif want_poster:
            lines.append("\n🖼️ 海报: 本地与联网都没有，可用 scripts/fetch_posters.py 补齐")

        if (detail or {}).get("votes"):
            lines.append(f"ℹ️ 豆瓣评价数: {detail['votes']}")
        return "\n".join(lines)
