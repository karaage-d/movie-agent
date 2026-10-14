# -*- coding: utf-8 -*-
"""电影智能体主入口。

用法：
  python main.py               启动网页版（默认）
  python main.py --clean       清洗数据
  python main.py --analyze     关联分析
  python main.py --build-kb    重建向量库
  python main.py --chat        命令行对话
"""
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Windows 控制台默认 GBK，日志与提示里有 emoji，直接 print 会抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 依赖（faiss-cpu 等）没有 3.15 的轮子，提前给出可读提示而不是让 pip/import 报错
if sys.version_info >= (3, 15):
    print("=" * 62)
    print(f"当前 Python 版本 {sys.version.split()[0]} 不受支持，请改用 Python 3.12。")
    print("faiss-cpu 最高支持到 3.14，zstandard 在 3.15 没有预编译包。")
    print()
    print("  py -3.12 -m venv .venv")
    print("  .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    print("  .venv\\Scripts\\python.exe main.py")
    print("=" * 62)
    sys.exit(1)

from src.utils import load_config, init_dirs


def cmd_clean(cfg):
    from src.data_cleaner import DataCleaner
    cleaner = DataCleaner(cfg)
    mdf, rdf = cleaner.process()
    print(f"\n✅ 清洗完成：{len(mdf)} 部电影, {len(rdf)} 条评论")


def cmd_analyze(cfg):
    from src.analyzer import MovieCorrelationAnalyzer
    analyzer = MovieCorrelationAnalyzer(cfg)
    results = analyzer.analyze()
    print("\n📊 分析结果：")
    for insight in results.get("insights", []):
        print(f"  {insight}")
    print(f"\n图表已保存到: {cfg['analysis']['output_dir']}")


def cmd_build_kb(cfg):
    from src.model_loader import load_llm, load_embedding
    from src.knowledge_base import MovieKnowledgeBase
    emb = load_embedding(cfg)
    kb = MovieKnowledgeBase(cfg, emb)
    kb.build(force_rebuild=True)
    print("✅ 知识库构建完成")


def cmd_chat(cfg):
    from src.model_loader import load_llm, load_embedding
    from src.knowledge_base import MovieKnowledgeBase
    from src.agent import MovieAgent
    print("🚀 正在加载模型和知识库...")
    llm = load_llm(cfg)
    emb = load_embedding(cfg)
    kb = MovieKnowledgeBase(cfg, emb).build()
    agent = MovieAgent(cfg, llm, kb)
    print("✅ 就绪！输入电影名开始聊天（输入 q 退出）\n")
    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("q", "quit", "exit", "退出"):
            break
        if not user:
            continue
        reply = agent.chat(user)
        print(f"\n影探: {reply}\n")


def cmd_web(cfg):
    """启动网页版：标准库 HTTP API + Vue 3 前端（无需 Node / 无需构建）"""
    from server.app import main as run_web
    run_web()


def main():
    parser = argparse.ArgumentParser(description="🎬 电影查询智能体")
    parser.add_argument("--web", action="store_true", help="启动网页版（默认）")
    parser.add_argument("--clean", action="store_true", help="清洗数据")
    parser.add_argument("--analyze", action="store_true", help="关联分析")
    parser.add_argument("--build-kb", action="store_true", help="构建知识库")
    parser.add_argument("--chat", action="store_true", help="命令行对话")
    args = parser.parse_args()

    cfg = load_config()
    init_dirs(cfg)

    if args.clean:
        cmd_clean(cfg)
    elif args.analyze:
        cmd_analyze(cfg)
    elif args.build_kb:
        cmd_build_kb(cfg)
    elif args.chat:
        cmd_chat(cfg)
    else:
        cmd_web(cfg)


if __name__ == "__main__":
    main()
