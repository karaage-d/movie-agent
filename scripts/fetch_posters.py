# -*- coding: utf-8 -*-
"""
补全缺失的本地海报（数据源：豆瓣公开 suggest 接口，无需 API Key）

用法（在项目根目录）：
    .venv\\Scripts\\python.exe scripts\\fetch_posters.py            # 只补缺的
    .venv\\Scripts\\python.exe scripts\\fetch_posters.py --limit 5  # 先试 5 部
    .venv\\Scripts\\python.exe scripts\\fetch_posters.py --retry    # 忽略已有大图，重下小图

做的事：
  1. 读 data/cleaned/movies.csv 的片名；
  2. 缺海报的逐个查豆瓣，把 CDN 小图 URL 升级成大图后下载到 data/posters/<片名>.jpg；
  3. 图片经过 PIL 校验（能打开、尺寸 >= 200px），不达标就丢弃；
  4. 把命中的海报 URL 回写到 movies.csv 的 poster_url 列，避免下次重复查询。
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):          # Windows 控制台默认 GBK，日志里有 emoji
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
from PIL import Image

from src.utils import load_config
from src.crawler import MovieCrawler

CSV_PATH = ROOT / "config" / "config.yaml"
MIN_SIDE = 200          # 最小边长，低于此值视为占位图/缩略图
MIN_BYTES = 3000

# 豆瓣搜索对这些片名命中不了（系列续作改名 / 译名不一致），人工给别名
ALIASES = {
    "哪吒之魔童闹海": ["哪吒2", "哪吒之魔童闹海 哪吒2", "哪吒2之魔童闹海"],
    "哪吒之魔童降世": ["哪吒之魔童降世 哪吒1", "哪吒降世"],
    "美国队长4：美丽新世界": ["美国队长4", "Captain America: Brave New World"],
    "流浪地球3": ["流浪地球3（上）", "流浪地球3 上"],
    "复仇者联盟4：终局之战": ["复仇者联盟4", "复仇者联盟4 终局之战"],
    "蝙蝠侠：黑暗骑士": ["蝙蝠侠：黑暗骑士 The Dark Knight", "黑暗骑士"],
    "指环王：护戒使者": ["指环王1：护戒使者", "指环王 护戒使者"],
    "这个杀手不太冷": ["这个杀手不太冷 Léon", "杀手莱昂"],
    "蛟龙行动": ["蛟龙行动 红海行动2", "红海行动2"],
}

# 豆瓣限流时会返回空数组（HTTP 200）。连续失败说明又快把自己打进限流了，
# 所以退避要够长，宁可慢一点也别触发风控。
BACKOFF_SECONDS = (15, 45, 90)
# 每成功处理 N 部插入一次长休息
BATCH_SIZE = 8
BATCH_REST_SECONDS = 20


def poster_path_for(crawler, title):
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)
    return crawler.posters_dir / f"{safe}.jpg"


def has_good_poster(crawler, title):
    p = poster_path_for(crawler, title)
    if not (p.exists() and p.stat().st_size > MIN_BYTES):
        return False
    try:
        with Image.open(p) as im:
            w, h = im.size
        return w >= MIN_SIDE and h >= MIN_SIDE
    except Exception:
        return False


def candidate_names(title):
    """依次尝试的片名：原片名 -> 去副标题 -> 人工别名"""
    seen = set()
    names = [title, re.split(r"[：:]", title)[0].strip()]
    names += ALIASES.get(title, [])
    for name in names:
        if name and name not in seen:
            seen.add(name)
            yield name


def search_with_backoff(crawler, name):
    """豆瓣限流时返回空数组，按 BACKOFF_SECONDS 递增等待重试"""
    for attempt, wait in enumerate((0,) + BACKOFF_SECONDS):
        if wait:
            time.sleep(wait)
        try:
            info = crawler._search_douban(name)
        except Exception:
            info = None
        if info and info.get("poster_url"):
            return info
        if attempt == len(BACKOFF_SECONDS):
            return None
    return None


def big_url(url):
    """豆瓣 CDN 的 s_/m_ 小图升级为 l_ 大图"""
    return re.sub(r"/(s|m)_ratio_poster/", "/l_ratio_poster/", url or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少部（0=全部）")
    ap.add_argument("--retry", action="store_true", help="忽略已有海报，强制重下")
    args = ap.parse_args()

    cfg = load_config()
    crawler = MovieCrawler(cfg)
    movies_csv = ROOT / cfg["data"]["movies_csv"]

    df = pd.read_csv(movies_csv, encoding="utf-8-sig")
    if "poster_url" not in df.columns:
        df["poster_url"] = ""
    # 全空列会被推断成 float64，直接写字符串会抛 LossySetitemError，先转成 object
    df["poster_url"] = df["poster_url"].astype("object").fillna("")

    titles = [str(t).strip() for t in df["title"].tolist()]
    todo = titles if args.retry else [t for t in titles if not has_good_poster(crawler, t)]
    if args.limit:
        todo = todo[: args.limit]

    print(f"电影库 {len(titles)} 部，本次待补 {len(todo)} 部\n")
    if not todo:
        print("✅ 全部都有可用海报了，无需处理")
        return

    done, failed = [], []
    for i, title in enumerate(todo, 1):
        got, source = None, ""
        for cand in candidate_names(title):
            info = search_with_backoff(crawler, cand)
            if info and info.get("poster_url"):
                raw = info["poster_url"]
                got = crawler.download_poster(big_url(raw), title)
                if got:
                    source = f"豆瓣命中「{info.get('title')}」({info.get('year')})"
                    break
            time.sleep(0.4)

        # 校验下载结果
        if got and not has_good_poster(crawler, title):
            try:
                os.remove(got)
            except OSError:
                pass
            got = None

        if got:
            with Image.open(got) as im:
                w, h = im.size
            # 记录本次实际生效的 URL（大图失败时 download_poster 会退回小图变体）
            record_url = big_url(raw)
            df.loc[df["title"].astype(str).str.strip() == title, "poster_url"] = record_url
            done.append(title)
            print(f"  [{i:>2}/{len(todo)}] {title:<20} ✅ {w}x{h}  {source}")
        else:
            failed.append(title)
            print(f"  [{i:>2}/{len(todo)}] {title:<20} ❌ 未找到可用海报")
        sys.stdout.flush()
        time.sleep(crawler.delay)
        if i % BATCH_SIZE == 0 and i < len(todo):
            print(f"  -- 已处理 {i} 部，休息 {BATCH_REST_SECONDS}s 避免触发豆瓣风控 --")
            sys.stdout.flush()
            time.sleep(BATCH_REST_SECONDS)

    # 回写 poster_url，避免下次重复查豆瓣
    df.to_csv(movies_csv, index=False, encoding="utf-8-sig")
    print(f"\n补齐 {len(done)} 部，失败 {len(failed)} 部")
    if failed:
        print("失败清单:", "、".join(failed))
    print(f"已回写 {movies_csv.name} 的 poster_url 列")


if __name__ == "__main__":
    main()
