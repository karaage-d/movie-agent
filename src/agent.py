# -*- coding: utf-8 -*-
"""
电影智能体核心
使用 LLM bind_tools + 手动调用循环，兼容所有 langchain 版本
"""
import yaml
import re
import json
import pandas as pd
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.utils import logger, ROOT_DIR, rating_label
from src.crawler import MovieCrawler

class MovieAgent:
    def __init__(self, cfg, llm, kb):
        self.cfg = cfg
        self.llm = llm
        self.kb = kb
        self.crawler = MovieCrawler(cfg)
        self.max_iterations = cfg["agent"]["max_iterations"]
        self.tools = []
        self.tool_map = {}
        # 多轮对话记忆：保存历史消息
        self.max_history_turns = int(cfg["agent"].get("max_history_turns", 10))
        self.history = []
        self._load_prompts()
        self._build_tools()
        self._build_agent()

    def _load_prompts(self):
        prompt_path = ROOT_DIR / "prompts" / "system_prompt.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)
        self.system_prompt = self.prompts["SYSTEM_PROMPT"]

    def _make_tool(self, func, name, description):
        """手动注册工具（不依赖 @tool 装饰器的版本差异）"""
        tool_def = {
            "name": name,
            "description": description,
            "parameters": self._infer_schema(func),
        }
        self.tools.append(tool_def)
        self.tool_map[name] = func
        return func

    def _infer_schema(self, func):
        """从函数签名推断 JSON Schema（简化版，够用就行）"""
        import inspect
        sig = inspect.signature(func)
        props = {}
        required = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            ptype = param.annotation
            pdesc = ""
            t = "string"
            if ptype == int or str(ptype) == "<class 'int'>":
                t = "integer"
            elif ptype == float:
                t = "number"
            elif ptype == bool:
                t = "boolean"
            props[pname] = {"type": t, "description": pdesc}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": props, "required": required}

    def _build_tools(self):
        kb = self.kb
        crawler = self.crawler
        cfg = self.cfg

        def search_movie_info(movie_name: str) -> str:
            """查询电影的基础信息：导演、主演、类型、年份、评分、剧情简介等。输入电影名称。"""
            info = kb.get_movie_info(movie_name)
            if not info:
                online = crawler.search_movie(movie_name)
                if online:
                    if online.get("poster_url"):
                        local_poster = crawler.download_poster(online["poster_url"], online["title"])
                        online["poster_local"] = local_poster or ""
                    return self._format_online_info(online)
                return f"抱歉，在知识库和网络上都没找到「{movie_name}」的信息，请检查电影名是否正确。"
            lines = [f"🎬 《{info.get('title','')}》({info.get('year','')})"]
            lines.append(f"🎥 导演: {info.get('director','未知')}")
            lines.append(f"⭐ 主演: {info.get('actors','未知')}")
            lines.append(f"🎭 类型: {info.get('genre','')}")
            lines.append(f"⏱️ 片长: {info.get('duration','')}分钟")
            if info.get("region"):
                lines.append(f"🌍 地区: {info.get('region','')}")
            dr = info.get("douban_rating")
            mr = info.get("maoyan_rating")
            dr_str = f"{dr}" if pd.notna(dr) else "暂无"
            mr_str = f"{mr}" if pd.notna(mr) else "暂无"
            lines.append(f"📊 豆瓣: {dr_str}  猫眼: {mr_str}")
            lines.append(f"\n📝 简介: {info.get('summary','暂无简介')}")
            return "\n".join(str(x) for x in lines if str(x) != "")

        def get_reviews(movie_name: str, count: int = 5) -> str:
            """获取电影的用户评论，区分好评和差评。输入电影名和返回条数。"""
            good = kb.get_movie_reviews(title=movie_name, sentiment="good", limit=count)
            bad = kb.get_movie_reviews(title=movie_name, sentiment="bad", limit=max(2, count//2))
            if len(good) == 0 and len(bad) == 0:
                return f"暂未找到《{movie_name}》的本地评论数据。"
            lines = [f"💬 《{movie_name}》观众评论精选："]
            if len(good) > 0:
                lines.append("\n👍 好评：")
                for _, r in good.iterrows():
                    lines.append(f"  · {r.get('username','用户')}（{r.get('rating','?')}分）：{r.get('content','')}")
            if len(bad) > 0:
                lines.append("\n👎 差评：")
                for _, r in bad.iterrows():
                    lines.append(f"  · {r.get('username','用户')}（{r.get('rating','?')}分）：{r.get('content','')}")
            return "\n".join(lines)

        def get_rating_and_good_rate(movie_name: str) -> str:
            """获取电影评分和好评率。输入电影名。"""
            info = kb.get_movie_info(movie_name)
            stats = kb.calc_good_rate(title=movie_name)
            if not info and stats["total"] == 0:
                online = crawler.search_movie(movie_name)
                if online and online.get("douban_rating"):
                    r = online["douban_rating"]
                    return f"📊 《{online.get('title', movie_name)}》\n豆瓣评分: {r} {rating_label(float(r))}\n（暂无本地评论统计）"
                return f"暂无《{movie_name}》的评分数据。"
            title = info.get("title", movie_name) if info else movie_name
            dr = info.get("douban_rating") if info else None
            lines = [f"📊 《{title}》评分概览："]
            if dr is not None and pd.notna(dr):
                lines.append(f"  豆瓣: {dr}  {rating_label(float(dr))}")
            if info and pd.notna(info.get("maoyan_rating")):
                lines.append(f"  猫眼: {info['maoyan_rating']}")
            if stats["total"] > 0:
                lines.append(
                    f"\n  样本评论 {stats['total']} 条："
                    f"好评 {stats['good']} 条 / 差评 {stats['bad']} 条"
                    f"\n  好评率: {stats['good_rate']}%"
                    f"\n  评论均分: {stats['avg_rating']}")
            return "\n".join(lines)

        def get_poster(movie_name: str) -> str:
            """获取电影海报图片。输入电影名。返回本地路径或在线图片URL。"""
            info = kb.get_movie_info(movie_name)
            poster_url = info.get("poster_url") if info else None
            if not poster_url or str(poster_url) in ["", "nan"]:
                online = crawler.search_movie(movie_name)
                if online and online.get("poster_url"):
                    title = online.get("title", movie_name)
                    local = crawler.download_poster(online["poster_url"], title)
                    if local:
                        return f"🎨 《{title}》海报路径: {local}"
                    else:
                        return f"🎨 《{title}》海报: {online['poster_url']}"
                return f"未找到《{movie_name}》的海报。"
            local = crawler.download_poster(poster_url, info["title"])
            if local:
                return f"🎨 《{info['title']}》海报路径: {local}"
            return f"🎨 《{info['title']}》海报: {poster_url}"

        def get_trailer(movie_name: str) -> str:
            """查找电影预告片链接，返回B站搜索链接或本地文件路径。输入电影名。"""
            # 1. 知识库有 trailer_url 直接用
            info = kb.get_movie_info(movie_name)
            if info:
                t = str(info.get("trailer_url") or "")
                if t and t not in ("nan", ""):
                    if t.startswith("http") or Path(t).exists():
                        return f"🎬 《{info['title']}》预告片: {t}"
            # 2. 在线搜索
            title = info.get("title", movie_name) if info else movie_name
            online_title = title
            if crawler.tmdb_key:
                online = crawler.search_movie(movie_name)
                if online and online.get("trailer_url"):
                    return f"🎬 《{online.get('title', title)}》预告片: {online['trailer_url']}"
                if online and online.get("title"):
                    online_title = online["title"]
            # 3. 兜底B站搜索
            url = crawler._bilibili_search_url(online_title)
            return f"🎬 《{online_title}》预告片: {url}"

        def recommend_similar(movie_name: str) -> str:
            """根据一部电影推荐相似电影。输入电影名。"""
            info = kb.get_movie_info(movie_name)
            if not info:
                return f"知识库中没有《{movie_name}》的信息，暂时无法推荐。"
            genre = str(info.get("genre", ""))
            director = str(info.get("director", ""))
            mpath = ROOT_DIR / cfg["data"]["movies_csv"]
            if not mpath.exists():
                return "暂无电影数据。"
            df = pd.read_csv(mpath, encoding="utf-8-sig")
            candidates = df[df["title"] != info["title"]]
            def match(row):
                g = str(row.get("genre", ""))
                d = str(row.get("director", ""))
                g_match = bool(set(g.replace(",", "/").split("/")) & set(genre.replace(",", "/").split("/"))) if genre else False
                d_match = bool(d and d != "nan" and director and director != "未知" and (d in director or director in d))
                return g_match or d_match
            mask = candidates.apply(match, axis=1)
            sim = candidates[mask].sort_values("douban_rating", ascending=False).head(3)
            if len(sim) == 0:
                sim = candidates.sort_values("douban_rating", ascending=False).head(3)
            lines = [f"🎯 如果你喜欢《{info['title']}》，推荐："]
            for _, row in sim.iterrows():
                lines.append(f"  · 《{row['title']}》({row.get('year','')}) — {row.get('genre','')}，豆瓣 {row.get('douban_rating','?')}")
            return "\n".join(lines)

        def search_nearby_cinemas(city: str, movie_name: str = "") -> str:
            """搜索附近电影院及排片信息。输入城市名和电影名（可选）。"""
            from src.cinema import search_nearby_cinemas as search_cinemas, format_cinemas_for_agent
            import os
            amap_key = (self.cfg.get("amap", {}).get("api_key", "")
                        or os.getenv("AMAP_API_KEY", ""))
            if not amap_key or amap_key == "YOUR_AMAP_KEY_HERE":
                return "⚠️ 未配置高德地图 API Key，请在 config/config.yaml 中填写 amap.api_key（免费注册：https://lbs.amap.com/）"
            cinemas = search_cinemas(amap_key, city, movie_name)
            return format_cinemas_for_agent(cinemas, city)

        self._make_tool(search_movie_info, "search_movie_info",
                        "查询电影的基础信息：导演、主演、类型、年份、评分、剧情简介等。参数: movie_name(电影名称)。")
        self._make_tool(get_reviews, "get_reviews",
                        "获取电影的用户评论，区分好评和差评。参数: movie_name(电影名), count(返回条数，默认5)。")
        self._make_tool(get_rating_and_good_rate, "get_rating_and_good_rate",
                        "获取电影评分和好评率。参数: movie_name(电影名)。")
        self._make_tool(get_poster, "get_poster",
                        "获取电影海报图片，成功返回本地图片路径。参数: movie_name(电影名)。")
        self._make_tool(get_trailer, "get_trailer",
                        "查找电影预告片链接。参数: movie_name(电影名)。")
        self._make_tool(recommend_similar, "recommend_similar",
                        "根据一部电影推荐相似电影。参数: movie_name(电影名)。")

        self._make_tool(search_nearby_cinemas, "search_nearby_cinemas",
                        "搜索附近电影院。参数: city(城市名), movie_name(电影名，可选)。")

        # ---- 列片单 & 联网兜底（解决"按条件找电影"和"库外新片"两类提问）----
        from src.agent_tools import MovieTools

        online = None
        try:
            from src.movie_online import MovieOnlineSource

            online = MovieOnlineSource(
                cfg, delay=float(cfg.get("crawler", {}).get("request_delay", 3) or 3))
        except Exception as exc:      # 联网源不可用不影响其余工具
            logger.warning(f"联网数据源初始化失败: {exc}")

        mt = MovieTools(cfg, kb, crawler=crawler, online=online)
        self.movie_tools = mt

        self._make_tool(mt.list_movies, "list_movies",
                        "按条件列出本地知识库里的电影片单。当用户问「有哪些电影」「XX年有什么新片」"
                        "「推荐几部科幻片」「评分8分以上的片子」这类**不针对某一部片**的问题时使用。"
                        "参数: year(年份或区间，如 2025 / 2020-2025，可留空), "
                        "genre(类型关键词，如 科幻，可留空), min_rating(最低豆瓣评分，0 表示不限), limit(返回条数)。")
        self._make_tool(mt.search_movie_web, "search_movie_web",
                        "查询一部电影的信息：**先查本地知识库，本地没有才联网爬取豆瓣实时数据**。"
                        "当用户提到具体片名、或本地 knowledge base 查不到该片时使用。"
                        "参数: movie_name(电影名称)。")

    def _build_agent(self):
        """绑定 tools 到 LLM"""
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        except Exception as e:
            logger.warning(f"bind_tools 失败({e})，回退到无工具模式")
            self.llm_with_tools = self.llm
        logger.info("✅ 电影智能体已就绪")

    def reset_history(self):
        """清空多轮对话历史"""
        self.history.clear()

    def _trim_history(self):
        while len(self.history) > self.max_history_turns * 2:
            self.history.pop(0)
            self.history.pop(0)

    def _parse_tool_calls(self, ai_msg):
        """从 AIMessage 中提取工具调用，兼容不同版本的格式"""
        tool_calls = []
        # 新版: additional_kwargs.tool_calls
        calls = getattr(ai_msg, "tool_calls", None)
        if calls:
            for c in calls:
                name = c.get("name")
                args = c.get("args", {})
                tool_id = c.get("id", "call_0")
                tool_calls.append({"name": name, "args": args, "id": tool_id})
            return tool_calls
        tc = ai_msg.additional_kwargs.get("tool_calls")
        if tc:
            for c in tc:
                fn = c.get("function", {})
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                tool_id = c.get("id", "call_0")
                tool_calls.append({"name": name, "args": args, "id": tool_id})
        return tool_calls

    def chat(self, user_input):
        self._trim_history()
        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.history)
        current_human = HumanMessage(content=user_input)
        messages.append(current_human)
        for i in range(self.max_iterations):
            try:
                ai_msg = self.llm_with_tools.invoke(messages)
            except Exception as e:
                return f"模型调用出错: {e}"

            tool_calls = self._parse_tool_calls(ai_msg)
            messages.append(ai_msg)

            if not tool_calls:
                final_text = ai_msg.content or "抱歉，我没能理解你的问题。"
                self.history.append(current_human)
                self.history.append(AIMessage(content=final_text))
                self._trim_history()
                return final_text

            # 执行工具调用
            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"] or {}
                tid = tc["id"]
                func = self.tool_map.get(name)
                if not func:
                    result = f"未知工具: {name}"
                else:
                    try:
                        # 把字符串数字参数转成正确类型
                        import inspect
                        sig = inspect.signature(func)
                        for pname, param in sig.parameters.items():
                            if pname in args and param.annotation == int and isinstance(args[pname], str):
                                try:
                                    args[pname] = int(args[pname])
                                except (ValueError, TypeError):
                                    pass
                        result = func(**args)
                    except Exception as e:
                        result = f"工具 {name} 执行出错: {e}"
                if self.cfg["agent"]["verbose"]:
                    logger.info(f"🔧 调用工具 {name}({args}) → {str(result)[:100]}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tid))

        return "抱歉，经过多轮推理仍未完成，请换个说法试试。"

    def chat_with_details(self, user_input):
        response = self.chat(user_input)
        poster_path = self._extract_path(response)
        trailer_url = self._extract_url(response)
        return {"answer": response, "poster_path": poster_path, "trailer_url": trailer_url}

    @staticmethod
    def _format_online_info(info):
        lines = [f"🎬 《{info.get('title','')}》({info.get('year','')})  [实时数据]"]
        if info.get("director"):
            lines.append(f"🎥 导演: {info['director']}")
        if info.get("actors"):
            lines.append(f"⭐ 主演: {info['actors']}")
        if info.get("genre"):
            lines.append(f"🎭 类型: {info['genre']}")
        if info.get("duration"):
            lines.append(f"⏱️ 片长: {info['duration']}分钟")
        if info.get("douban_rating"):
            lines.append(f"📊 评分: {info['douban_rating']} {rating_label(float(info['douban_rating']))}")
        if info.get("summary"):
            lines.append(f"\n📝 简介: {info['summary']}")
        return "\n".join(lines)

    @staticmethod
    def _extract_path(text):
        # 优先匹配本地路径
        m = re.search(r'([A-Za-z]:\\[^\s<]+|/[^\s<]+\.(?:jpg|jpeg|png|webp))', text)
        if m:
            return m.group(1)
        # 再匹配URL
        m = re.search(r'(https?://[^\s<]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s<]*)?)', text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_url(text):
        # 匹配 http(s) 链接，支持中文、+、?、&、=、% 等，遇空白或中文句号截断
        m = re.search(r'https?://[^\s<>"\'\u201c\u201d\u2018\u2019\u300c\u300d\u300a\u300b]+', text)
        return m.group(0) if m else None