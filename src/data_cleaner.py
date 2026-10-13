# -*- coding: utf-8 -*-
"""
数据清洗与预处理模块
"""
import re
import pandas as pd
from pathlib import Path
from src.utils import logger, ROOT_DIR, ensure_dir


class DataCleaner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_dir = ROOT_DIR / cfg["data"]["raw_dir"]
        self.cleaned_dir = ROOT_DIR / cfg["data"]["cleaned_dir"]
        ensure_dir(self.cleaned_dir)

    def process(self, raw_movies_file=None, raw_reviews_file=None):
        logger.info("=" * 50)
        logger.info("开始数据清洗与预处理...")
        movies_df = self._load_and_clean_movies(raw_movies_file)
        reviews_df = self._load_and_clean_reviews(raw_reviews_file, movies_df)
        movies_path = self.cleaned_dir / "movies.csv"
        reviews_path = self.cleaned_dir / "reviews.csv"
        movies_df.to_csv(movies_path, index=False, encoding="utf-8-sig")
        reviews_df.to_csv(reviews_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 清洗完成: {len(movies_df)} 部电影, {len(reviews_df)} 条评论")
        return movies_df, reviews_df

    def _load_and_clean_movies(self, raw_file=None):
        files = self._find_raw_files(raw_file, ["movies", "movie", "film"])
        if not files:
            logger.warning("未找到电影原始数据，将使用示例数据")
            return self._create_sample_movies()
        dfs = []
        for f in files:
            try:
                df = self._read_file(f)
                dfs.append(df)
                logger.info(f"  读取: {f.name} ({len(df)} 行)")
            except Exception as e:
                logger.warning(f"  读取失败 {f.name}: {e}")
        if not dfs:
            return self._create_sample_movies()
        df = pd.concat(dfs, ignore_index=True)
        return self._clean_movies(df)

    def _clean_movies(self, df):
        col_map = {
            "电影名": "title", "电影名称": "title", "片名": "title", "name": "title", "title": "title",
            "导演": "director", "directors": "director",
            "主演": "actors", "演员": "actors", "cast": "actors",
            "类型": "genre", "genres": "genre",
            "年份": "year", "上映年份": "year", "release_year": "year",
            "上映日期": "release_date",
            "豆瓣评分": "douban_rating", "评分": "rating", "rating": "douban_rating",
            "猫眼评分": "maoyan_rating",
            "IMDb评分": "imdb_rating",
            "片长": "duration", "时长": "duration", "runtime": "duration",
            "简介": "summary", "剧情简介": "summary", "description": "summary",
            "海报URL": "poster_url", "海报": "poster_url", "poster": "poster_url",
            "预告片URL": "trailer_url", "预告片": "trailer_url", "trailer": "trailer_url",
            "国家/地区": "region", "地区": "region", "country": "region",
            "语言": "language",
            "票房": "box_office",
        }
        df = df.rename(columns={c: col_map.get(c.strip(), c) for c in df.columns})
        for col in ["title"]:
            if col not in df.columns:
                raise ValueError(f"数据缺少必需列: {col}")
        if "year" in df.columns:
            df = df.drop_duplicates(subset=["title", "year"], keep="first")
        else:
            df = df.drop_duplicates(subset=["title"], keep="first")
        df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
        for col in ["douban_rating", "maoyan_rating", "imdb_rating"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[(df[col] < 0) | (df[col] > 10), col] = None
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        text_cols = ["title", "director", "actors", "genre", "summary"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self._clean_text)
        df["summary"] = df.get("summary", "").fillna("暂无简介")
        df["director"] = df.get("director", "").fillna("未知")
        df["actors"] = df.get("actors", "").fillna("未知")
        df["genre"] = df.get("genre", "").fillna("其他")
        df.insert(0, "movie_id", range(1, len(df) + 1))
        return df.reset_index(drop=True)

    def _load_and_clean_reviews(self, raw_file=None, movies_df=None):
        files = self._find_raw_files(raw_file, ["reviews", "review", "comments", "comment", "评论"])
        if not files:
            logger.warning("未找到评论原始数据，将使用示例评论")
            return self._create_sample_reviews(movies_df)
        dfs = []
        for f in files:
            try:
                df = self._read_file(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"  读取失败 {f.name}: {e}")
        if not dfs:
            return self._create_sample_reviews(movies_df)
        df = pd.concat(dfs, ignore_index=True)
        return self._clean_reviews(df, movies_df)

    def _clean_reviews(self, df, movies_df=None):
        col_map = {
            "电影名": "title", "电影名称": "title", "片名": "title",
            "用户名": "username", "用户": "username", "user": "username",
            "评论内容": "content", "评论": "content", "内容": "content",
            "评分": "rating", "打分": "rating",
            "评论时间": "date", "时间": "date",
            "点赞数": "votes", "有用数": "votes",
        }
        df = df.rename(columns={c: col_map.get(c.strip(), c) for c in df.columns})
        if "title" not in df.columns or "content" not in df.columns:
            return pd.DataFrame(columns=["review_id", "movie_id", "title", "username", "content", "rating", "date", "votes"])
        df = df.drop_duplicates(subset=["content"], keep="first")
        df = df[df["content"].notna()]
        df = df[df["content"].astype(str).str.len() > 5]
        df["content"] = df["content"].astype(str).apply(self._clean_text)
        df["title"] = df["title"].astype(str).apply(self._clean_text)
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df.loc[(df["rating"] < 0) | (df["rating"] > 10), "rating"] = None
        if "votes" in df.columns:
            df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
        if movies_df is not None and len(movies_df) > 0:
            title2id = dict(zip(movies_df["title"], movies_df["movie_id"]))
            df["movie_id"] = df["title"].map(title2id)
        else:
            df["movie_id"] = None
        df.insert(0, "review_id", range(1, len(df) + 1))
        return df.reset_index(drop=True)

    def _find_raw_files(self, raw_file, keywords):
        if raw_file:
            p = Path(raw_file)
            if not p.is_absolute():
                p = self.raw_dir / p
            return [p] if p.exists() else []
        files = []
        if self.raw_dir.exists():
            for f in self.raw_dir.iterdir():
                if f.suffix.lower() in [".csv", ".xlsx", ".xls", ".json"]:
                    if any(k in f.stem.lower() for k in keywords):
                        files.append(f)
        return files

    def _read_file(self, f):
        suffix = f.suffix.lower()
        if suffix == ".csv":
            for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
                try:
                    return pd.read_csv(f, encoding=enc)
                except UnicodeDecodeError:
                    continue
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(f)
        elif suffix == ".json":
            return pd.read_json(f)
        raise ValueError(f"不支持的文件格式: {suffix}")

    @staticmethod
    def _clean_text(text):
        if not isinstance(text, str):
            return text
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _create_sample_movies(self):
        data = [
            [1, "流浪地球2", "郭帆", "吴京/刘德华/李雪健", "科幻/灾难/冒险", 2023, 8.3, 9.4, 8.0,
             173, "太阳即将毁灭，人类在地球表面建造出巨大的推进器，寻找新的家园。然而宇宙之路危机四伏，为了拯救地球，流浪地球时代的年轻人再次挺身而出。",
             "", "", "中国大陆", "汉语普通话/英语/俄语", "40.29亿"],
            [2, "满江红", "张艺谋", "沈腾/易烊千玺/张译", "剧情/喜剧/悬疑/古装", 2023, 7.0, 9.1, 7.0,
             159, "南宋绍兴年间，岳飞死后四年，秦桧率兵与金国会谈。会谈前夜，金国使者死在宰相驻地，所携密信也不翼而飞。",
             "", "", "中国大陆", "汉语普通话", "45.44亿"],
            [3, "奥本海默", "克里斯托弗·诺兰", "基里安·墨菲/艾米莉·布朗特", "剧情/传记/历史", 2023, 8.9, 8.8, 8.4,
             180, "讲述美国原子弹之父罗伯特·奥本海默主导制造出人类第一颗原子弹的故事。",
             "", "", "美国/英国", "英语", "4.5亿"],
            [4, "封神第一部：朝歌风云", "乌尔善", "费翔/李雪健/黄渤/于适", "动作/战争/奇幻/古装", 2023, 7.8, 9.1, 7.0,
             148, "商王殷寿勾结狐妖妲己，暴虐无道，引发天谴。昆仑仙人姜子牙携封神榜下山，寻找天下共主。",
             "", "", "中国大陆", "汉语普通话", "26.34亿"],
            [5, "芭比", "格蕾塔·葛韦格", "玛格特·罗比/瑞恩·高斯林", "喜剧/奇幻/冒险", 2023, 8.0, 7.8, 6.8,
             114, "芭比在芭比乐园里过着完美的生活，直到她开始思考存在的意义，前往现实世界寻找真相。",
             "", "", "美国/英国", "英语", "2.5亿"],
            [6, "消失的她", "崔睿/刘翔", "朱一龙/倪妮/文咏珊", "悬疑/犯罪", 2023, 6.5, 9.2, 6.4,
             122, "何非的妻子李木子在结婚周年旅行中离奇消失，而一位自称是他妻子的女子出现在他身边。",
             "", "", "中国大陆", "汉语普通话", "35.23亿"],
            [7, "长安三万里", "谢君伟/邹靖", "杨天翔/凌振赫（配音）", "动画/历史", 2023, 8.3, 9.4, 7.5,
             168, "安史之乱爆发后数年，吐蕃大军攻打西南。大唐节度使高适交战不利，向监军太监回忆起与李白的一生往事。",
             "", "", "中国大陆", "汉语普通话", "18.24亿"],
            [8, "蜘蛛侠：纵横宇宙", "华金·多斯·桑托斯", "沙梅克·摩尔/海莉·斯坦菲尔德（配音）", "动画/动作/科幻/冒险", 2023, 8.5, 8.9, 8.6,
             140, "小黑蛛迈尔斯·莫拉莱斯与格温·史黛西重聚，二人穿梭多元宇宙，遭遇蜘蛛侠联盟。",
             "", "", "美国", "英语", "3.5亿"],
            [9, "孤注一掷", "申奥", "张艺兴/金晨/咏梅/王传君", "剧情/犯罪", 2023, 6.9, 9.1, 6.7,
             130, "程序员潘生和模特梁安娜被海外高薪招聘吸引，出国淘金，却意外落入境外诈骗工厂的陷阱。",
             "", "", "中国大陆", "汉语普通话", "38.47亿"],
            [10, "热辣滚烫", "贾玲", "贾玲/雷佳音", "喜剧/剧情", 2024, 7.8, 9.3, 7.0,
             126, "宅家多年的乐莹，在一场家庭变故后，决定为自己拼一次，走上了拳击之路。",
             "", "", "中国大陆", "汉语普通话", "34.6亿"],
            [11, "哪吒之魔童闹海", "饺子", "吕艳婷/囧森瑟夫/瀚墨（配音）", "动画/奇幻/动作", 2025, 8.5, 9.7, 7.8,
             144, "天劫之后，哪吒、敖丙的灵魂虽保住了，但肉身很快会魂飞魄散。太乙真人打算用七色宝莲给二人重塑肉身，但是在重塑过程中却遇到了前所未有的困难，哪吒的命运又将何去何从。",
             "", "", "中国大陆", "汉语普通话", "150亿+"],
            [12, "唐探1900", "陈思诚/戴墨", "王宝强/刘昊然/周润发", "喜剧/悬疑/动作", 2025, 6.8, 9.0, 6.5,
             135, "1900年的旧金山，唐人街发生连环命案，刚到美国的华裔青年秦福与当地华人阿鬼联手探案，却意外卷入一场涉及排华法案的巨大阴谋。",
             "", "", "中国大陆", "汉语普通话/英语", "32亿+"],
            [13, "蛟龙行动", "林超贤", "黄轩/于适/杜江/蒋璐霞", "动作/战争", 2025, 7.2, 9.1, 6.8,
             131, "伊维亚内战刚结束，蛟龙战队接到新任务，潜入深海执行秘密营救行动，面对恐怖组织的新型武器威胁，一场高燃的深海对决即将展开。",
             "", "", "中国大陆", "汉语普通话", "28亿+"],
            [14, "美国队长4：美丽新世界", "朱利叶斯·约拿", "安东尼·麦凯/哈里森·福特", "动作/科幻/冒险", 2025, 5.8, 7.8, 5.5,
             118, "山姆·威尔逊正式接任美国队长，在与新总统罗斯的交涉中，意外发现一场针对全球超级英雄的阴谋，以色列首领萨宾·阿尔塞蒙德将掀起新的世界大战。",
             "", "", "美国", "英语", "3.2亿"],
            [15, "好东西", "邵艺辉", "钟楚曦/章宇/宋佳/赵又廷", "剧情/喜剧/爱情", 2024, 9.1, 9.0, 7.5,
             123, "单亲妈妈王铁梅带着女儿和母亲生活，在邻居小马的影响下开始重新审视自我和女性的生存处境，一段温暖又尖锐的都市女性故事。",
             "", "", "中国大陆", "汉语普通话", "7.2亿"],
            [16, "抓娃娃", "闫非/彭大魔", "沈腾/马丽/史彭元", "喜剧/剧情", 2024, 7.1, 9.2, 6.6,
             133, "一对夫妻为了让儿子继承家业，在家中布下天罗地网，用二十年时间精心设计了一场沉浸式'穷养'骗局，结果却被儿子反将一军。",
             "", "", "中国大陆", "汉语普通话", "33亿"],
            [17, "流浪地球3", "郭帆", "吴京/刘德华/李雪健/沈腾", "科幻/灾难/冒险", 2025, 8.0, 9.5, 7.6,
             155, "面对太阳氦闪危机，人类分为流浪地球派与数字生命派两大阵营，联合政府遭遇政变，刘培强和图恒宇必须联手化解这场关乎人类文明存亡的危机。",
             "", "", "中国大陆", "汉语普通话", "45亿+"],
        ]
        columns = ["movie_id", "title", "director", "actors", "genre", "year",
                   "douban_rating", "maoyan_rating", "imdb_rating", "duration",
                   "summary", "poster_url", "trailer_url", "region", "language", "box_office"]
        return pd.DataFrame(data, columns=columns)

    def _create_sample_reviews(self, movies_df):
        sample_reviews = {
            "流浪地球2": [
                ("科幻迷小王", "中国科幻的里程碑！特效不输好莱坞，剧情比第一部更扎实，刘德华的线太催泪了。", 9, 2023),
                ("路人甲", "三个小时有点长，中间部分节奏拖了，但是整体还是值得一看。", 7, 2023),
                ("差评君", "剧情有bug，很多逻辑不通顺，科学设定有点硬伤。", 5, 2023),
            ],
            "满江红": [
                ("喜剧控", "沈腾和易烊千玺化学反应太棒了！反转再反转，最后全军复诵满江红那段直接泪崩。", 8, 2023),
                ("历史爱好者", "作为悬疑片还行，但别当历史片看，改编太多。", 6, 2023),
            ],
            "奥本海默": [
                ("诺兰粉", "诺兰神作！视听震撼，墨菲演技封神，IMAX看是享受。", 9, 2023),
                ("普通观众", "台词太密，人物多，不了解物理史的会有点跟不上。", 7, 2023),
            ],
            "哪吒之魔童闹海": [
                ("国漫真爱粉", "饺子导演太牛了！特效国漫天花板，剧情紧凑催泪，最后那段直接泪崩！", 10, 2025),
                ("带娃家长", "孩子看得很开心，大人也觉得不错，就是结尾彩蛋太多等了十分钟。", 8, 2025),
                ("理性观众", "画面确实顶级，但剧情相比第一部稍显套路，期待第三部。", 7, 2025),
            ],
            "唐探1900": [
                ("喜剧迷", "王宝强还是那个味儿！周润发的加盟太惊喜了，旧金山的年代感很足。", 8, 2025),
                ("推理控", "推理部分弱了些，更偏喜剧冒险，不过春节档看还是很欢乐。", 6, 2025),
            ],
            "蛟龙行动": [
                ("军事迷", "林超贤的战争戏永远在线！海底作战那段太刺激了，音效炸裂。", 9, 2025),
                ("普通观众", "节奏比红海行动稍微慢一点，但整体场面还是值回票价。", 7, 2025),
            ],
            "好东西": [
                ("豆瓣文青", "邵艺辉太会写了！女性视角细腻又犀利，台词句句扎心，年度最佳！", 10, 2024),
                ("男性观众", "被女朋友拉去看的，意外地好看，宋佳和钟楚曦演得真好。", 8, 2024),
            ],
            "抓娃娃": [
                ("开心麻花粉", "沈腾马丽组合永远不让人失望，反转不断，全场爆笑。", 9, 2024),
                ("评论家", "喜剧外壳下是教育观的反思，笑点和立意都有，但中后段稍显拖沓。", 7, 2024),
            ],
            "流浪地球3": [
                ("科幻迷", "郭帆又一次刷新国产科幻上限！数字生命派vs流浪派的对决太震撼，期待第四部。", 9, 2025),
                ("路人观众", "沈腾的加入让笑点多了很多，但三个小时有点坐不住。", 7, 2025),
            ],
        }
        rows = []
        rid = 1
        title2id = dict(zip(movies_df["title"], movies_df["movie_id"])) if movies_df is not None else {}
        for title, reviews in sample_reviews.items():
            for username, content, rating, year in reviews:
                rows.append({
                    "review_id": rid,
                    "movie_id": title2id.get(title),
                    "title": title,
                    "username": username,
                    "content": content,
                    "rating": rating,
                    "date": f"{year}-02-15",
                    "votes": 100 + rid * 10,
                })
                rid += 1
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.utils import load_config, init_dirs
    cfg = load_config()
    init_dirs(cfg)
    cleaner = DataCleaner(cfg)
    mdf, rdf = cleaner.process()
    print(mdf[["title", "douban_rating"]])
