# -*- coding: utf-8 -*-
"""
模型加载器：统一加载 LLM 和 Embedding 模型

LLM 支持：通义千问(qwen) / DeepSeek / OpenAI / 本地 Ollama
Embedding 支持：通义在线(text-embedding-v3) / 本地 BGE / M3E

qwen、DeepSeek、OpenAI 三家都走 OpenAI 兼容接口，共用同一套自研 BaseChatModel 封装
（含 bind_tools 与 tool_calls 编解码），因此换模型只需改 config.yaml，无需改代码。
"""
import os

from src.utils import logger, setup_model_cache

# 本地向量模型（bge/m3e）会把模型缓存到 HF_HOME；必须在导入
# huggingface_hub / sentence_transformers 之前锁定到项目内的 models_cache/
setup_model_cache()

# ---- 各家的 OpenAI 兼容端点 ----
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _env(name):
    """读环境变量，空字符串视为未设置"""
    return os.getenv(name) or ""



def load_llm(cfg):
    """根据 config.yaml 的 model.provider 加载对应的 LLM"""
    m = cfg["model"]
    provider = m["provider"].lower()
    temperature = m["temperature"]
    max_tokens = m["max_tokens"]

    # ---------------- 通义千问 ----------------
    if provider == "qwen":
        api_key = m.get("qwen_api_key") or _env("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "请在 config/config.yaml 中填写 model.qwen_api_key，"
                "或设置环境变量 DASHSCOPE_API_KEY\n"
                "申请地址: https://dashscope.console.aliyun.com/"
            )
        model_name = m.get("qwen_model", "qwen-plus")
        logger.info(f"加载通义千问模型: {model_name}")
        return _make_openai_chat(
            api_key, model_name, temperature, max_tokens,
            base_url=QWEN_BASE_URL,
            # 通义专属：开启模型侧联网搜索，用于覆盖知识库之外的新片
            extra_body={
                "enable_search": True,
                "search_options": {
                    "search_enabled": True,
                    "search_result_count": 5,
                    "forced_search": False,
                },
            },
        )

    # ---------------- DeepSeek ----------------
    elif provider == "deepseek":
        api_key = m.get("deepseek_api_key") or _env("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "请在 config/config.yaml 中填写 model.deepseek_api_key，"
                "或设置环境变量 DEEPSEEK_API_KEY\n"
                "申请地址: https://platform.deepseek.com/api_keys"
            )
        model_name = m.get("deepseek_model", "deepseek-flash")
        base_url = m.get("deepseek_base_url") or DEEPSEEK_BASE_URL
        thinking = bool(m.get("deepseek_thinking", False))
        logger.info(f"加载 DeepSeek 模型: {model_name}（思考模式: {'开启' if thinking else '关闭'}）")
        # DeepSeek 的思考模式默认开启；本项目是多轮工具调用场景，
        # 关闭后响应更快、更省 token。需要深度推理时把 deepseek_thinking 设为 true。
        return _make_openai_chat(
            api_key, model_name, temperature, max_tokens,
            base_url=base_url,
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )

    # ---------------- OpenAI ----------------
    elif provider == "openai":
        api_key = m.get("openai_api_key") or _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "请在 config/config.yaml 中填写 model.openai_api_key，"
                "或设置环境变量 OPENAI_API_KEY"
            )
        model_name = m.get("openai_model", "gpt-4o-mini")
        base_url = m.get("openai_base_url") or OPENAI_BASE_URL
        logger.info(f"加载 OpenAI 模型: {model_name}")
        return _make_openai_chat(
            api_key, model_name, temperature, max_tokens, base_url=base_url,
        )

    # ---------------- 本地 Ollama ----------------
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        logger.info(f"加载本地Ollama模型: {m['ollama_model']}")
        return ChatOllama(
            model=m["ollama_model"],
            base_url=m["ollama_base_url"],
            temperature=temperature,
            num_predict=max_tokens,
        )

    else:
        raise ValueError(
            f"不支持的模型提供商: {provider}，可选 qwen / deepseek / openai / ollama"
        )


def _make_openai_chat(api_key, model_name, temperature, max_tokens, base_url,
                      extra_body=None):
    """用 OpenAI 兼容接口调用各家模型（通义千问 / DeepSeek / OpenAI 通用）

    自研封装的原因：第三方 langchain-dashscope 等适配包与 LangChain 版本频繁冲突；
    自己实现 BaseChatModel 接口后，只要对方兼容 OpenAI 协议就能接入。
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    import json

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 库: pip install openai")

    fixed_extra_body = dict(extra_body or {})

    class OpenAICompatChatModel(BaseChatModel):
        api_key_: str = ""
        model_name_: str = ""
        base_url_: str = ""
        temperature_: float = 0.3
        max_tokens_: int = 2048
        _tools: list = []
        _extra_body: dict = {}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.api_key_ = api_key
            self.model_name_ = model_name
            self.base_url_ = base_url
            self.temperature_ = temperature
            self.max_tokens_ = max_tokens
            self._tools = []
            self._extra_body = dict(fixed_extra_body)
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        @property
        def _llm_type(self):
            return "openai-compatible-chat"

        def _convert_messages(self, messages):
            msgs = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    msgs.append({"role": "system", "content": m.content})
                elif isinstance(m, HumanMessage):
                    msgs.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    if m.tool_calls:
                        tool_calls = []
                        for tc in m.tool_calls:
                            tool_calls.append({
                                "id": tc.get("id", "call_0"),
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["args"], ensure_ascii=False) if not isinstance(tc["args"], str) else tc["args"],
                                }
                            })
                        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": tool_calls})
                    else:
                        msgs.append({"role": "assistant", "content": m.content or ""})
                elif hasattr(m, "tool_call_id"):
                    msgs.append({"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id})
                else:
                    msgs.append({"role": "user", "content": str(m.content)})
            return msgs

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            msgs = self._convert_messages(messages)
            tools = kwargs.get("tools", None) or getattr(self, "_tools", None)

            call_kwargs = dict(
                model=self.model_name_,
                messages=msgs,
                temperature=self.temperature_,
                max_tokens=self.max_tokens_,
            )

            if tools:
                call_kwargs["tools"] = tools
            # 各家专属参数：通义用 enable_search 开启联网；DeepSeek 用 thinking 控制思考模式
            if self._extra_body:
                call_kwargs["extra_body"] = dict(self._extra_body)

            resp = self._client.chat.completions.create(**call_kwargs)
            choice = resp.choices[0]
            msg = choice.message
            content = msg.content or ""

            # 解析 tool_calls
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({
                        "name": tc.function.name,
                        "args": args,
                        "id": tc.id,
                    })

            ai_msg = AIMessage(content=content, tool_calls=tool_calls if tool_calls else [])
            gen = ChatGeneration(message=ai_msg)
            return ChatResult(generations=[gen])

        def bind_tools(self, tools):
            """把工具定义转成 OpenAI Function Calling 格式（各家通用）"""
            api_tools = []
            for t in tools:
                tdef = {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    }
                }
                api_tools.append(tdef)

            new_model = OpenAICompatChatModel()
            new_model._tools = api_tools
            # 新实例不会自动继承 extra_body，必须显式复制，
            # 否则 bind_tools 之后通义的联网搜索 / DeepSeek 的思考模式开关会丢失
            new_model._extra_body = dict(self._extra_body)
            return new_model

    return OpenAICompatChatModel()


def load_embedding(cfg):
    """加载Embedding模型"""
    provider = cfg["embedding"]["provider"].lower()

    if provider == "qwen":
        api_key = cfg["model"].get("qwen_api_key") or _env("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "向量化需要通义的 API Key（DeepSeek 不提供 Embedding 接口）：\n"
                "  方案一：在 config/config.yaml 填写 model.qwen_api_key"
                "（已有 data/vector_db/ 时不会实际发起调用）\n"
                "  方案二：把 embedding.provider 改成 bge，改用本地模型"
                "（需先安装 sentence-transformers）"
            )
        logger.info("使用通义千问在线Embedding (text-embedding-v3)")
        return _make_qwen_embedding(api_key, cfg["embedding"].get("model_name", "text-embedding-v3"))

    elif provider == "bge":
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        except ImportError:
            from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
        logger.info(f"加载BGE Embedding: {cfg['embedding']['model_name']}")
        return HuggingFaceBgeEmbeddings(
            model_name=cfg["embedding"]["model_name"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=cfg["embedding"]["cache_dir"],
        )

    elif provider == "m3e":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
        logger.info(f"加载M3E Embedding: {cfg['embedding']['model_name']}")
        return HuggingFaceEmbeddings(
            model_name=cfg["embedding"]["model_name"],
            cache_folder=cfg["embedding"]["cache_dir"],
        )

    else:
        raise ValueError(f"不支持的Embedding: {provider}，可选 qwen/bge/m3e")


def _make_qwen_embedding(api_key, model_name):
    """用原生 dashscope SDK 包装 Embedding，避免 langchain_dashscope 版本问题"""
    from langchain_core.embeddings import Embeddings
    import dashscope
    from dashscope import TextEmbedding

    class QwenEmbeddings(Embeddings):
        api_key_: str
        model_: str

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.api_key_ = api_key
            self.model_ = model_name
            dashscope.api_key = api_key

        def embed_documents(self, texts):
            result = []
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                resp = TextEmbedding.call(
                    model=self.model_,
                    input=batch,
                    dimensions=1024,
                    api_key=self.api_key_,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"通义Embedding调用失败: {resp.code} - {resp.message}")
                result.extend([e["embedding"] for e in resp.output["embeddings"]])
            return result

        def embed_query(self, text):
            resp = TextEmbedding.call(
                model=self.model_,
                input=[text],
                dimensions=1024,
                api_key=self.api_key_,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"通义Embedding调用失败: {resp.code} - {resp.message}")
            return resp.output["embeddings"][0]["embedding"]

    return QwenEmbeddings()
