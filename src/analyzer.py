# -*- coding: utf-8 -*-
"""
电影数据关联分析模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import logger, ROOT_DIR, ensure_dir

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 深色主题，与网页端界面一致，避免白底图表突兀
DARK_BG = "#0d1016"
DARK_FG = "#b3bbc9"
DARK_TITLE = "#f4f6fa"
DARK_GRID = "#232a36"
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.edgecolor": "#2a3140",
    "axes.labelcolor": DARK_FG,
    "text.color": DARK_TITLE,
    "xtick.color": DARK_FG,
    "ytick.color": DARK_FG,
    "grid.color": DARK_GRID,
    "legend.facecolor": DARK_BG,
    "legend.edgecolor": "#2a3140",
})


class MovieCorrelationAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = ROOT_DIR / cfg["analysis"]["output_dir"]
        self.method = cfg["analysis"]["method"]
        self.threshold = 0.7
        self.plot = cfg["analysis"]["plot"]
        ensure_dir(self.output_dir)

    def analyze(self, movies_csv=None):
        mpath = Path(movies_csv) if movies_csv else ROOT_DIR / self.cfg["data"]["movies_csv"]
        if not mpath.exists():
            logger.warning(f"电影数据不存在: {mpath}")
            return {}
        df = pd.read_csv(mpath, encoding="utf-8-sig")
        logger.info(f"开始关联分析，共 {len(df)} 部电影")

        results = {
            "numeric_corr": self._numeric_correlation(df),
            "genre_analysis": self._genre_analysis(df),
            "top_directors": self._director_analysis(df),
            "strong_correlations": [],
            "insights": [],
        }
        corr_matrix = results["numeric_corr"]
        if not corr_matrix.empty:
            cols = corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr_matrix.iloc[i, j]
                    if pd.notna(val) and abs(val) >= self.threshold:
                        results["strong_correlations"].append({
                            "var1": cols[i], "var2": cols[j],
                            "correlation": round(float(val), 3),
                            "direction": "正相关" if val > 0 else "负相关",
                        })
        results["insights"] = self._generate_insights(results)
        if self.plot:
            self._plot_genre_box(df)
            logger.info(f"分析图表已保存到 {self.output_dir}")
        return results

    def _numeric_correlation(self, df):
        numeric_cols = []
        for col in ["douban_rating", "maoyan_rating", "imdb_rating", "duration", "year"]:
            if col in df.columns:
                numeric_cols.append(col)
        if "box_office" in df.columns:
            df = df.copy()
            df["box_office_num"] = df["box_office"].apply(self._parse_box_office)
            numeric_cols.append("box_office_num")
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return num_df.corr(method=self.method).round(3)

    @staticmethod
    def _parse_box_office(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        try:
            if "亿" in s:
                return float(s.replace("亿", "").strip())
            if "万" in s:
                return float(s.replace("万", "").strip()) / 10000
            return float(s)
        except ValueError:
            return np.nan

    def _genre_analysis(self, df):
        if "genre" not in df.columns or "douban_rating" not in df.columns:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            genres = str(row["genre"]).replace(",", "/").split("/")
            for g in genres:
                g = g.strip()
                if g and g != "nan":
                    records.append({"genre": g, "rating": row["douban_rating"]})
        gdf = pd.DataFrame(records)
        if gdf.empty:
            return pd.DataFrame()
        agg = gdf.groupby("genre")["rating"].agg(count="count", avg_rating="mean", std="std").round(2)
        return agg.sort_values("avg_rating", ascending=False).reset_index()

    def _director_analysis(self, df, top_n=10):
        if "director" not in df.columns or "douban_rating" not in df.columns:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            directors = str(row["director"]).replace(",", "/").split("/")
            for d in directors:
                d = d.strip()
                if d and d not in ["未知", "nan"]:
                    records.append({"director": d, "rating": row["douban_rating"]})
        ddf = pd.DataFrame(records)
        if ddf.empty:
            return pd.DataFrame()
        agg = ddf.groupby("director")["rating"].agg(count="count", avg_rating="mean").round(2)
        agg = agg[agg["count"] >= 1].sort_values("avg_rating", ascending=False)
        return agg.head(top_n).reset_index()

    def _generate_insights(self, results):
        insights = []
        for item in results["strong_correlations"]:
            insights.append(
                f"{item['var1']} 与 {item['var2']} 呈显著{item['direction']}（相关系数 {item['correlation']}）")
        ga = results["genre_analysis"]
        if not ga.empty and len(ga) > 0:
            top_g = ga.iloc[0]
            worst_g = ga.iloc[-1]
            insights.append(
                f"平均分最高的类型是「{top_g['genre']}」（{top_g['avg_rating']}分，{int(top_g['count'])}部）；"
                f"最低的是「{worst_g['genre']}」（{worst_g['avg_rating']}分）")
        da = results["top_directors"]
        if not da.empty and len(da) > 0:
            insights.append(f"样本中评分最高的导演是「{da.iloc[0]['director']}」（平均{da.iloc[0]['avg_rating']}分）")
        if not insights:
            insights.append("数据量较小，暂未发现显著关联，建议扩充数据集后再分析。")
        return insights

    def _plot_genre_box(self, df):
        if "genre" not in df.columns or "douban_rating" not in df.columns:
            return
        records = []
        for _, row in df.iterrows():
            genres = str(row["genre"]).replace(",", "/").split("/")
            for g in genres:
                g = g.strip()
                if g and g != "nan":
                    records.append({"genre": g, "rating": row["douban_rating"]})
        gdf = pd.DataFrame(records)
        if gdf.empty:
            return
        order = gdf.groupby("genre")["rating"].median().sort_values(ascending=False).index.tolist()
        # 类型多时不要把画布拉得过宽，否则在网页通栏展示时标签会被缩得太小
        width = min(14.5, max(9.0, len(order) * 0.62))
        fig, ax = plt.subplots(figsize=(width, 5.6))
        sns.boxplot(data=gdf, x="genre", y="rating", order=order, ax=ax, palette="Set2",
                    linewidth=1.1, fliersize=3,
                    boxprops={"alpha": 0.92},
                    medianprops={"color": DARK_TITLE, "linewidth": 1.6})
        ax.set_title("各电影类型评分分布", fontsize=14, pad=15, color=DARK_TITLE)
        ax.set_xlabel("类型", color=DARK_FG)
        ax.set_ylabel("豆瓣评分", color=DARK_FG)
        ax.grid(axis="y", color=DARK_GRID, linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#2a3140")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(self.output_dir / "genre_rating_box.png", dpi=150,
                    bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
