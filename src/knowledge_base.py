# -*- coding: utf-8 -*-
"""
知识库构建 & RAG检索模块
"""
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.utils import logger, ROOT_DIR, ensure_dir


class MovieKnowledgeBase:
    def __init__(self, cfg, embedding_model):
        self.cfg = cfg
        self.embedding = embedding_model
        self.persist_dir = ROOT_DIR / cfg["vectorstore"]["persist_dir"]
        self.vectorstore = None
        self.movies_df = None
        self.reviews_df = None
        self.chunk_size = 500
        self.chunk_overlap = 50
        ensure_dir(self.persist_dir)

    def build(self, movies_csv=None, reviews_csv=None, force_rebuild=False):
        if not force_rebuild and self._load_existing():
            # 顺便加载csv
            mpath = ROOT_DIR / self.cfg["data"]["movies_csv"]
            rpath = ROOT_DIR / self.cfg["data"]["reviews_csv"]
            if mpath.exists():
                self.movies_df = pd.read_csv(mpath, encoding="utf-8-sig")
            if rpath.exists():
                self.reviews_df = pd.read_csv(rpath, encoding="utf-8-sig")
            logger.info("加载已有知识库")
            return self

        mpath = Path(movies_csv) if movies_csv else ROOT_DIR / self.cfg["data"]["movies_csv"]
        rpath = Path(reviews_csv) if reviews_csv else ROOT_DIR / self.cfg["data"]["reviews_csv"]
        if not mpath.exists():
            raise FileNotFoundError(f"电影数据不存在: {mpath}，请先运行数据清洗")
        self.movies_df = pd.read_csv(mpath, encoding="utf-8-sig")
        self.reviews_df = pd.read_csv(rpath, encoding="utf-8-sig") if rpath.exists() else pd.DataFrame()

        logger.info(f"开始构建知识库: {len(self.movies_df)} 部电影, {len(self.reviews_df)} 条评论")
        docs = self._build_documents()
        logger.info(f"生成 {len(docs)} 个文档片段")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"分块后: {len(chunks)} 个文本块")
        logger.info("正在向量化（首次会下载BGE模型，请耐心等待）...")
        self.vectorstore = FAISS.from_documents(chunks, self.embedding)
        self.vectorstore.save_local(str(self.persist_dir))
        logger.info(f"✅ 知识库构建完成，已保存到 {self.persist_dir}")
        return self

    def _load_existing(self):
        index_file = self.persist_dir / "index.faiss"
        if index_file.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.persist_dir), self.embedding,
                    allow_dangerous_deserialization=True,
                )
                return True
            except Exception as e:
                logger.warning(f"加载已有知识库失败: {e}")
        return False

    def _build_documents(self):
        docs = []
        for _, row in self.movies_df.iterrows():
            content = self._format_movie_doc(row)
            metadata = {
                "type": "movie",
                "movie_id": int(row.get("movie_id", 0)),
                "title": str(row.get("title", "")),
                "year": str(row.get("year", "")),
                "genre": str(row.get("genre", "")),
                "director": str(row.get("director", "")),
                "douban_rating": float(row["douban_rating"]) if pd.notna(row.get("douban_rating")) else None,
            }
            docs.append(Document(page_content=content, metadata=metadata))

        if self.reviews_df is not None and len(self.reviews_df) > 0:
            for _, row in self.reviews_df.iterrows():
                title = str(row.get("title", ""))
                content = f"电影《{title}》用户评论：{row.get('content', '')}"
                metadata = {
                    "type": "review",
                    "movie_id": int(row["movie_id"]) if pd.notna(row.get("movie_id")) else None,
                    "title": title,
                    "rating": float(row["rating"]) if pd.notna(row.get("rating")) else None,
                    "username": str(row.get("username", "")),
                }
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    @staticmethod
    def _format_movie_doc(row):
        parts = [f"电影《{row.get('title','')}》基本信息："]
        fields = [
            ("导演", "director"), ("主演", "actors"), ("类型", "genre"),
            ("年份", "year"), ("片长(分钟)", "duration"), ("地区", "region"),
            ("语言", "language"), ("豆瓣评分", "douban_rating"),
            ("猫眼评分", "maoyan_rating"), ("IMDb评分", "imdb_rating"),
            ("票房", "box_office"),
        ]
        for label, key in fields:
            v = row.get(key)
            if pd.notna(v) and str(v).strip() not in ["", "未知", "nan"]:
                parts.append(f"- {label}: {v}")
        summary = row.get("summary", "")
        if pd.notna(summary) and str(summary).strip() not in ["", "暂无简介", "nan"]:
            parts.append(f"\n剧情简介：{summary}")
        return "\n".join(parts)

    def search(self, query, k=5, filter_type=None):
        if self.vectorstore is None:
            if not self._load_existing():
                raise RuntimeError("知识库未构建，请先调用 build()")
        if filter_type:
            results = self.vectorstore.similarity_search(query, k=k * 3)
            results = [d for d in results if d.metadata.get("type") == filter_type][:k]
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        return results

    def get_movie_info(self, title):
        if self.movies_df is None:
            mpath = ROOT_DIR / self.cfg["data"]["movies_csv"]
            if mpath.exists():
                self.movies_df = pd.read_csv(mpath, encoding="utf-8-sig")
            else:
                return None
        matches = self.movies_df[self.movies_df["title"].str.contains(title, na=False, case=False)]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()

    def get_movie_reviews(self, title=None, movie_id=None, sentiment=None, limit=10):
        if self.reviews_df is None:
            rpath = ROOT_DIR / self.cfg["data"]["reviews_csv"]
            if rpath.exists():
                self.reviews_df = pd.read_csv(rpath, encoding="utf-8-sig")
            else:
                return pd.DataFrame()
        df = self.reviews_df
        if title:
            df = df[df["title"].str.contains(title, na=False, case=False)]
        if movie_id:
            df = df[df["movie_id"] == movie_id]
        if sentiment == "good":
            df = df[df["rating"] >= self.cfg["agent"]["good_rating_threshold"]]
        elif sentiment == "bad":
            df = df[df["rating"] < self.cfg["agent"]["good_rating_threshold"]]
        return df.head(limit)

    def calc_good_rate(self, title=None, movie_id=None):
        df = self.get_movie_reviews(title=title, movie_id=movie_id, limit=9999)
        if len(df) == 0:
            return {"total": 0, "good": 0, "bad": 0, "good_rate": 0.0, "avg_rating": 0.0}
        threshold = self.cfg["agent"]["good_rating_threshold"]
        good = (df["rating"] >= threshold).sum()
        bad = len(df) - good
        return {
            "total": int(len(df)),
            "good": int(good),
            "bad": int(bad),
            "good_rate": round(good / len(df) * 100, 1),
            "avg_rating": round(df["rating"].mean(), 2),
        }
